import utils_misc
from tools import bool_parse

freer_gpu = utils_misc.select_freer_gpu()

import argparse
import os
import socket
import time

import numpy as np
import pandas as pd
import wandb
from datasets import Dataset

from evaluation import evaluate_model
from utils_dataset import cc_newsela_collate, cc_news_collate, cnn_dailymail_collate

from torch.utils.data import DataLoader, RandomSampler

from model_salience import CoverageModel
from model_fluency import FluencyRelativeScore, TextDiscriminator
from model_simplicity import SimplicitySyntacticScore, SimplicityLexicalScore
from model_guardrails import (
    RelativeBrevityPenalizer,
    NERInaccuracyPenalty,
    RepeatNGramPenalty,
)

import utils_optim, utils_scoring, utils_rl, utils_timing
from model_generator import Generator
from datasets import load_dataset
from datetime import datetime
import torch

from transformers import logging

logging.set_verbosity_error()

try:
    from torch.cuda import amp

    use_torch_amp = True
except ImportError:
    use_torch_amp = False

n = 500

parser = argparse.ArgumentParser()
parser.add_argument(
    "experiment",
    type=str,
    help="Experiment name. Will be used to save a model file and a log file.",
)
parser.add_argument(
    "coverage_model_path",
    type=str,
    help="Coverage model bin file path.",
)

# Generator
parser.add_argument(
    "model_start_file",
    type=str,
    help="Starting model file of the generator.",
)

parser.add_argument(
    "ckpt_output_path",
    type=str,
    help="Checkpoint output file path to use to export checkpoint.",
)

parser.add_argument(
    "--model_card",
    type=str,
    default="gpt2-medium",
    help="What folder contains the model configuration.",
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=90,
    help="Maximum output length. Saves time if the sequences are short.",
)
parser.add_argument(
    "--num_runs",
    type=int,
    default=8,
    help="For each element in the batch, how many independent runs to have; The `k` of k-SCST",
)
parser.add_argument(
    "--scoring",
    type=str,
    default="product",
    choices=["product", "logsum"],
    help="The way individual scores are aggregated into a total score. Can be `product` or `logsum`",
)

# Optimization
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
parser.add_argument(
    "--optimizer_name",
    type=str,
    default="adam",
    choices=["adam", "lamb"],
    help="Two options for now: `adam` and `lamb`",
)
parser.add_argument(
    "--train_batch_size", type=int, default=1, help="Training batch size."
)

# CKPT Related
parser.add_argument(
    "--ckpt_every",
    type=int,
    default=600,
    help="If 0, checkpointing is not used. Otherwise, checkpointing is done very x seconds.",
)
parser.add_argument(
    "--ckpt_lookback",
    type=int,
    default=100,
    help="When checkpointing, will consider the avg total score of the last x samples.",
)
parser.add_argument(
    "--print_every",
    type=int,
    default=150,
    help="Save the model and print out an example every x seconds.",
)
parser.add_argument(
    "--timings",
    type=bool_parse,
    default=True,
    help="Whether to print out timings for each pass.",
)

# Dataset
parser.add_argument(
    "--dataset", choices=["cc_news", "cnn_dailymail"], type=str, default="cc_news"
)
parser.add_argument("--max_steps", type=int, default="50000")

args = parser.parse_args()

args.experiment += "_" + socket.gethostname()

wandb.init(project="keep_it_simple")
wandb.config.update(args)
wandb.run.name = args.experiment

timing = args.timings
utils_misc.DoublePrint(
    "simplifier_%s_%s.log" % (args.experiment, datetime.now().strftime("%Y_%m_%d")),
    show_timings=timing,
)

N_samples = args.num_runs
max_seq_length = args.max_seq_length
simplifier = Generator(
    args.model_card,
    seq2seq=("gpt2" not in args.model_card),
    max_input_length=max_seq_length,
    max_output_length=max_seq_length,
    device="cuda",
)
simplifier.reload(args.model_start_file)
simplifier.eval()

train_batch_size = args.train_batch_size
if args.dataset == "cc_news":
    train_dataset = load_dataset(args.dataset, split="train")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=RandomSampler(train_dataset),
        drop_last=True,
        collate_fn=cc_news_collate,
    )
elif args.dataset == "cnn_dailymail":
    train_dataset = load_dataset(args.dataset, split="train")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=RandomSampler(train_dataset),
        drop_last=True,
        collate_fn=cnn_dailymail_collate,
    )
else:
    raise ValueError(f"Dataset {args.dataset} not supported.")

# The val dataset to eval after each 10K steps + at zero steps
dataset_df = pd.read_csv(os.path.join("datastore", "newsela_paired_0.2.csv"))
val_dataset_df = dataset_df[dataset_df["cut"] == "train"]
val_dataset = Dataset.from_pandas(val_dataset_df)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=train_batch_size,
    sampler=RandomSampler(val_dataset),
    drop_last=True,
    collate_fn=cc_newsela_collate,
)

# The test dataset use as the final evaluation of the model
test_dataset_df = dataset_df[dataset_df["cut"] == "dev"]
test_dataset = Dataset.from_pandas(test_dataset_df)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=train_batch_size,
    sampler=RandomSampler(test_dataset),
    drop_last=True,
    collate_fn=cc_newsela_collate,
    pin_memory=True,
)

optimizer = utils_optim.build_optimizer(
    simplifier.model,
    optimizer_name=args.optimizer_name,
    learning_rate=args.learning_rate,
)

ckpter = utils_rl.RLModelCheckpoint(
    simplifier,
    args.ckpt_every,
    args.ckpt_lookback,
    os.path.join(args.ckpt_output_path, args.experiment + ".bin"),
)
printer = utils_rl.RLExamplePrinter(
    args.print_every, N_samples, print_source=False, print_edit=True
)
timer = utils_timing.TickTimer()
thermostat = utils_rl.RLThermostat()
rl_crit = utils_rl.ReinforceCriterion(simplifier, optimizer, use_apex=use_torch_amp)

coverage_model = CoverageModel(
    "nostop",
    model_file=args.coverage_model_path,
    fp16=True,
    is_soft=True,
)

scorers = [
    {
        "name": "coverage",
        "model": coverage_model,
        "sign": 1,
        "weight": 2.0,
    },
    {
        "name": "simple_syn",
        "model": SimplicitySyntacticScore(),
        "sign": 1,
        "weight": 4.0,
    },
    {
        "name": "simple_lex",
        "model": SimplicityLexicalScore(target_shift=0.4, word_change_ratio=0.15),
        "sign": 1,
        "weight": 2.0,
    },
    {"name": "fluency_lm", "model": FluencyRelativeScore(), "sign": 1},
    {
        "name": "fluency_disc",
        "model": TextDiscriminator(retrain_every=800, fp16=True),
        "sign": 1,
    },
    {
        "name": "gr_repeat",
        "model": RepeatNGramPenalty(gram=3),
        "sign": -1,
        "weight": 2.0,
    },
    {
        "name": "gr_brevity",
        "model": RelativeBrevityPenalizer(min_ratio=0.6, max_ratio=1.3),
        "sign": -1,
        "weight": 2.0,
    },
    {
        "name": "gr_hallucination",
        "model": NERInaccuracyPenalty(),
        "sign": -1,
        "weight": 2.0,
    },
]

scorer = utils_scoring.ScorerWrapper(
    scorers, scoring_method=args.scoring, max_batch_size=12
)
T_start, T_last_best = time.time(), time.time()
temperature = 1.0

max_steps = args.max_steps
eval_frequency = 5
for idx, paragraphs in enumerate(train_dataloader):
    gene_params = {
        "max_output_length": max_seq_length,
        "sample": True,
        "num_runs": N_samples,
        "no_repeat_ngram": 5,
        "max_batch_size": 12,
        "no_copy_ngram": 7,
        "temperature": temperature,
    }

    if idx == 0:
        print("--- Doing evaluation of the model on the val set ---")
        scores = evaluate_model(
            model=simplifier,
            coverage_model=coverage_model,
            dataloader=val_dataloader,
            n=n,
        )

        eval_log = {f"val/{k}": v for k, v in scores.items()}
        wandb.log(
            eval_log, commit=False
        )  # commit=false does not increment Steps in log

    # We do max_steps steps
    if idx < max_steps:
        T_batch_start = time.time()

        # Doing real, sampled generation
        gens_out = simplifier.generate(paragraphs, **gene_params)
        unlooped_batch = [
            {
                "paragraph": p,
                "generated": gen["output_text"],
                "generated_tokenized": gen["output_tokens"],
            }
            for p, gens in zip(paragraphs, gens_out)
            for gen in gens
        ]
        unlooped_paragraphs = [d["paragraph"] for d in unlooped_batch]
        generateds = [d["generated"] for d in unlooped_batch]
        generateds_tokenized = [d["generated_tokenized"] for d in unlooped_batch]
        timer.tick("sampled_generation")

        scorer_returns = scorer.score(unlooped_paragraphs, generateds)

        # total_scores are the RS_j value i.e. the product of the rewards terms as per article.
        RS_j = torch.FloatTensor(scorer_returns["total_scores"]).cuda()

        batch_RS_j = RS_j.reshape(train_batch_size, N_samples)
        R_Overline_S = torch.repeat_interleave(batch_RS_j.mean(dim=1), N_samples)

        timer.tick("all_scores")
        # The first loss term is the R Overline S minus RS_j (see equation 5 in article).
        first_loss_term = R_Overline_S - batch_RS_j
        n_diff_pos, n_diff_neg = (first_loss_term < -0.02).long().sum().item(), (
            first_loss_term > 0.02
        ).long().sum().item()
        print(
            "[%d steps out of %d] [%d samples] %d above avg and %d below avg with a 0.02 margin."
            % (idx, max_steps, train_batch_size * N_samples, n_diff_pos, n_diff_neg)
        )

        diversity = len(set(generateds)) / len(generateds)
        temperature = thermostat.log_diversity(diversity)
        loss = rl_crit(unlooped_paragraphs, generateds_tokenized, first_loss_term)
        timer.tick("optim")

        batch_time = time.time() - T_batch_start
        log_obj = {
            "train/loss": loss,
            "train/max_scores": torch.max(RS_j),
            "train/temperature": temperature,
            "train/elem_per_sec": (len(generateds) / (batch_time + 0.001)),
        }
        log_obj.update(
            {
                f"train/mean_{k}": np.mean(v)
                for k, v in scorer_returns.items()
                if "_scores" in k or k in ["fluency_disc_val_f1"]
            }
        )
        log_obj.update(
            {
                f"train/{k}": v
                for k, v in scorer_returns.items()
                if "_scores" in k or k in ["fluency_disc_val_f1"]
            }
        )
        log_obj.update(
            {
                "train/coverage_original_sentence": scorer_returns[
                    "coverage_original_sentence"
                ][0],
                "train/coverage_all_masked_words_in_sentence": "; ".join(
                    scorer_returns["coverage_all_masked_words_in_sentences"][0]
                ),
                "train/coverage_effective_mask_ratio": scorer_returns[
                    "coverage_effective_mask_ratios"
                ][0],
            }
        )
        if timing:
            timer.report()

        # Run the Checkpoint engine
        current_score = np.mean(scorer_returns["total_scores"])
        is_best = ckpter.tick(current_score)
        if is_best:  # Run the inspection dataset through
            T_last_best = time.time()

        # Run the Printing engine
        printer.tick(paragraphs, generateds, scorer_returns)

        # Since each Wandb.log increase the step, we log the training with the eval to better align results
        if (idx % eval_frequency) == 0 and idx > 0:
            print("--- Doing evaluation of the model on the val set ---")
            scores = evaluate_model(
                model=simplifier,
                coverage_model=coverage_model,
                dataloader=val_dataloader,
                n=n,
            )

            log_obj.update({f"val/{k}": v for k, v in scores.items()})
            wandb.log(log_obj)

            if (idx + 1) == 500:
                # The first 500 steps, we evaluate it each five steps
                # Then, for the steps between 500 and 10000, we evaluate it each 100 steps
                # Thus, we raise the eval_frequency
                eval_frequency = 100
            elif (idx + 1) == 10000:
                # Then, for the steps between 10 000 and max_steps, we evaluate it each 10 000 steps
                eval_frequency = 10000
        else:
            wandb.log(log_obj)
    else:
        break

print("--- Doing evaluation of the model on the test set ---")
scores = evaluate_model(
    model=simplifier, coverage_model=coverage_model, dataloader=test_dataloader, n=n
)

# commit=false does not increment Steps in log
test_log_obj = {
    "test/average_sari_score": scores["average_sari_score"],
    "test/average_bleu_score": scores["average_bleu_score"],
    "test/fkgl_ratio_score": scores["fkgl_ratio_score"],
    "test/compression_rate_score": scores["compression_rate_score"],
    "test/coverage_rate_score": scores["coverage_rate_score"],
}

wandb.log(test_log_obj, commit=False)
