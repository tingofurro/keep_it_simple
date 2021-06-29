import utils_misc, time, argparse, numpy as np, wandb
freer_gpu = utils_misc.select_freer_gpu()

from torch.utils.data import DataLoader, RandomSampler
from apex import amp

from model_salience import CoverageModel
from model_fluency import FluencyRelativeScore, TextDiscriminator
from model_simplicity import SimplicitySyntacticScore, SimplicityLexicalScore
from model_guardrails import RelativeBrevityPenalizer, NERInaccuracyPenalty, RepeatNGramPenalty

import utils_optim, utils_scoring, utils_rl, utils_timing
from model_generator import Generator
from datasets import load_dataset
from datetime import datetime
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True, help="Experiment name. Will be used to save a model file and a log file.")

# Generator
parser.add_argument("--model_card", type=str, default="gpt2-medium", help="What folder contains the model configuration.")
parser.add_argument("--model_start_file", required=True, type=str, help="Starting model file of the generator.")
parser.add_argument("--max_seq_length", type=int, default=90, help="Maximum output length. Saves time if the sequences are short.")
parser.add_argument("--num_runs", type=int, default=10, help="For each element in the batch, how many independent runs to have; The `k` of k-SCST")
parser.add_argument("--scoring", type=str, default="logsum", help="The way individual scores are aggregated into a total score. Can be `product` or `logsum`")

# Optimization
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--optimizer_name", type=str, default="lamb", help="Two options for now: `adam` and `lamb`")
parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size.")

# CKPT Related
parser.add_argument("--ckpt_every", type=int, default=600, help="If 0, checkpointing is not used. Otherwise, checkpointing is done very x seconds.")
parser.add_argument("--ckpt_lookback", type=int, default=100, help="When checkpointing, will consider the avg total score of the last x samples.")
parser.add_argument("--print_every", type=int, default=150, help="Save the model and print out an example every x seconds.")
parser.add_argument("--timings", action='store_true', help="Whether to print out timings for each pass.")

args = parser.parse_args()

args.experiment += "_"+freer_gpu

wandb.init(project="keep_it_simple")
wandb.config.update(args)
wandb.run.name = args.experiment
wandb.run.save()
utils_misc.DoublePrint("simplifier_%s_%s.log" % (args.experiment, datetime.now().strftime("%Y_%m_%d")), show_timings=True)

N_samples = args.num_runs
simplifier = Generator(args.model_card, seq2seq=("gpt2" not in args.model_card), max_input_length=args.max_seq_length, max_output_length=args.max_seq_length, device="cuda")
simplifier.reload(args.model_start_file)
simplifier.eval()

def cc_news_collate(inps):
    batch_paras = []
    for inp in inps:
        text = inp["text"]
        paragraphs = sorted(text.split("\n"), key=lambda p: abs(p.count(" ")-35))
        batch_paras.append(paragraphs[0])
    return batch_paras


dataset = load_dataset("cc_news")["train"]
dataloader = DataLoader(dataset=dataset, batch_size=args.train_batch_size, sampler=RandomSampler(dataset), drop_last=True, collate_fn=cc_news_collate)
optimizer = utils_optim.build_optimizer(simplifier.model, optimizer_name=args.optimizer_name, learning_rate=args.learning_rate)

use_apex = True
if use_apex:
    simplifier.model, optimizer = amp.initialize(simplifier.model, optimizer, opt_level="O2") # O1 is really not good, according to experiments on 10/13/2020

ckpter = utils_rl.RLModelCheckpoint(simplifier, args.ckpt_every, args.ckpt_lookback, "/home/phillab/models/simplifier/%s.bin" % (args.experiment))
printer = utils_rl.RLExamplePrinter(args.print_every, N_samples, print_source=False, print_edit=True)
timer = utils_timing.TickTimer()
thermostat = utils_rl.RLThermostat()
rl_crit = utils_rl.ReinforceCriterion(simplifier, optimizer, use_apex=use_apex)

scorers = [{"name": "coverage", "model": CoverageModel("nostop", model_file="/home/phillab/models/coverage_roberta.bin", fp16=True, is_soft=True), "sign": 1, "weight": 2.0},
           {"name": "simple_syn", "model": SimplicitySyntacticScore(), "sign": 1, "weight": 4.0},
           {"name": "simple_lex", "model": SimplicityLexicalScore(target_shift=0.4, word_change_ratio=0.15), "sign": 1, "weight": 2.0},
           {"name": "fluency_lm", "model": FluencyRelativeScore(), "sign": 1},
           {"name": "fluency_disc", "model": TextDiscriminator(retrain_every=800, fp16=True), "sign": 1},
           {"name": "gr_repeat", "model": RepeatNGramPenalty(gram=3), "sign": -1, "weight": 2.0},
           {"name": "gr_brevity", "model": RelativeBrevityPenalizer(min_ratio=0.6, max_ratio=1.3), "sign": -1, "weight": 2.0},
           {"name": "gr_hallucination", "model": NERInaccuracyPenalty(), "sign": -1, "weight": 2.0}
           ]

scorer = utils_scoring.ScorerWrapper(scorers, scoring_method=args.scoring, max_batch_size=12)
T_start, T_last_best = time.time(), time.time()
temperature = 1.0

for epoch_i in range(30):
    for ib, paragraphs in enumerate(dataloader):
        T_batch_start = time.time()
        gene_params = {"max_output_length": args.max_seq_length, "sample": True, "num_runs": args.num_runs, "no_repeat_ngram": 5, "max_batch_size": 12, "no_copy_ngram": 7, "temperature": temperature}

        # Doing real, sampled generation
        gens_out = simplifier.generate(paragraphs, **gene_params)
        unlooped_batch = [{"paragraph": p, "generated": gen['output_text'], "generated_tokenized": gen['output_tokens']} for p, gens in zip(paragraphs, gens_out) for gen in gens]
        unlooped_paragraphs = [d['paragraph'] for d in unlooped_batch]
        generateds = [d['generated'] for d in unlooped_batch]
        generateds_tokenized = [d['generated_tokenized'] for d in unlooped_batch]
        timer.tick("sampled_generation")

        scorer_returns = scorer.score(unlooped_paragraphs, generateds)
        total_scores = torch.FloatTensor(scorer_returns["total_scores"]).cuda()

        batch_scores = total_scores.reshape(args.train_batch_size, N_samples)
        mean_scores = batch_scores.mean(dim=1)
        max_scores = torch.max(batch_scores, dim=1).values # For logging
        unlooped_mean_scores = torch.repeat_interleave(mean_scores, N_samples)

        timer.tick("all_scores")
        normalized_rewards = (unlooped_mean_scores - total_scores)
        n_diff_pos, n_diff_neg = (normalized_rewards<-0.02).long().sum().item(), (normalized_rewards>0.02).long().sum().item()
        print("[%d samples] %d above avg and %d below avg" % (args.train_batch_size*N_samples, n_diff_pos, n_diff_neg))

        diversity = len(set(generateds)) / len(generateds)
        temperature = thermostat.log_diversity(diversity)
        loss = rl_crit(unlooped_paragraphs, generateds_tokenized, normalized_rewards)
        timer.tick("optim")

        batch_time = time.time() - T_batch_start
        log_obj = {"loss": loss, "max_scores": max_scores.mean().item(), "temperature": temperature, "elem_per_sec": (len(generateds) / (batch_time+0.001))}
        log_obj.update({k: np.mean(v) for k, v in scorer_returns.items() if "_scores" in k or k in ["fluency_disc_val_f1"]})
        wandb.log(log_obj)

        if args.timings:
            timer.report()

        # Run the Checkpoint engine
        current_score = np.mean(scorer_returns['total_scores'])
        is_best = ckpter.tick(current_score)
        if is_best: # Run the inspection dataset through
            T_last_best = time.time()

        # Run the Printing engine
        printer.tick(paragraphs, generateds, scorer_returns)
