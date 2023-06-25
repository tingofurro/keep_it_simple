from functools import partial

import utils_misc
from tools import bool_parse
from utils_fluency import pre_process_min_max_fluency_log_probs
from utils_train import create_comparison_tag

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
from utils_dataset import (
    keyed_collate_fn,
)

from torch.utils.data import DataLoader, random_split

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
    "--dataset",
    choices=["cc_news", "cnn_dailymail", "xsum", "imdb"],
    type=str,
    default="cc_news",
)
parser.add_argument("--max_steps", type=int, default="40000")
parser.add_argument("--n_eval", type=int, default="500")
parser.add_argument(
    "--include_original",
    type=bool_parse,
    default=False,
    help="Whether to include the original sentence in the sampled sentence.",
)
parser.add_argument("--compute_eval_lexile", type=bool_parse, default=False)
parser.add_argument("--reward_components_weights", type=bool_parse, default=True)
parser.add_argument("--fluency_min_max", type=bool_parse, default=False)


args = parser.parse_args()

experiment_name = args.experiment + "_" + socket.gethostname()

comparison_tag = create_comparison_tag(args)

vars(args).update({"tag": comparison_tag})

wandb.init(project="keep_it_simple")
wandb.config.update(args)
wandb.run.name = experiment_name

timing = args.timings
utils_misc.DoublePrint(
    "simplifier_%s_%s.log" % (experiment_name, datetime.now().strftime("%Y_%m_%d")),
    show_timings=timing,
)

include_original = args.include_original
n_samples = args.num_runs

n_eval = args.n_eval

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
dataset_name = args.dataset
if dataset_name == "cc_news":
    train_dataset = load_dataset(dataset_name, split="train")
    train_collate_fn = partial(keyed_collate_fn, key="text")
elif dataset_name == "cnn_dailymail":
    train_dataset = load_dataset(dataset_name, "3.0.0", split="train")
    train_collate_fn = partial(keyed_collate_fn, key="article")
elif dataset_name == "xsum":
    train_dataset = load_dataset(dataset_name, split="train")
    train_collate_fn = partial(keyed_collate_fn, key="document")
elif dataset_name == "imdb":
    train_dataset = load_dataset(dataset_name, split="unsupervised")
    train_collate_fn = partial(keyed_collate_fn, key="text")
else:
    raise ValueError(f"Dataset {dataset_name} not supported.")

max_steps = args.max_steps

# We split the dataset into train and val subset of length max_steps and the rest of the dataset.
train_indices_max_steps, _ = random_split(
    train_dataset, [max_steps, len(train_dataset) - max_steps]
)
train_split_max_steps = train_dataset.select(train_indices_max_steps.indices)

train_dataloader = DataLoader(
    dataset=train_split_max_steps,
    batch_size=train_batch_size,
    drop_last=True,
    collate_fn=train_collate_fn,
)

n_eval = args.n_eval

newsela_collate = partial(keyed_collate_fn, key="p1")
# The val dataset to eval at specific steps
dataset_df = pd.read_csv(os.path.join("datastore", "newsela_paired_0.2.csv"))
val_dataset_df = dataset_df[dataset_df["cut"] == "train"]
val_dataset = Dataset.from_pandas(val_dataset_df)

# We split the dataset into val and empty subset of length n_eval and the rest of the dataset.
val_indices_n_eval, _ = random_split(val_dataset, [n_eval, len(val_dataset) - n_eval])
val_split_n_eval = val_dataset.select(val_indices_n_eval.indices)

val_dataloader = DataLoader(
    dataset=val_split_n_eval,
    batch_size=train_batch_size,
    drop_last=True,
    collate_fn=newsela_collate,
)

# The test dataset use as the final evaluation of the model
test_dataset_df = dataset_df[dataset_df["cut"] == "dev"]
test_dataset = Dataset.from_pandas(test_dataset_df)

# We split the dataset into test and empty of length n_eval and the rest of the dataset.
test_indices_n_eval, _ = random_split(
    test_dataset, [n_eval, len(test_dataset) - n_eval]
)
test_split_n_eval = test_dataset.select(test_indices_n_eval.indices)

test_dataloader = DataLoader(
    dataset=test_split_n_eval,
    batch_size=train_batch_size,
    drop_last=True,
    collate_fn=newsela_collate,
)

optimizer = utils_optim.build_optimizer(
    simplifier.model,
    optimizer_name=args.optimizer_name,
    learning_rate=args.learning_rate,
)

output_dir_path = os.path.join(args.ckpt_output_path, dataset_name)
os.makedirs(output_dir_path, exist_ok=True)

ckpter = utils_rl.RLModelCheckpoint(
    simplifier,
    args.ckpt_every,
    args.ckpt_lookback,
    os.path.join(output_dir_path, experiment_name + ".bin"),
)

save_path = os.path.join(output_dir_path, experiment_name + ".txt")
if os.path.exists(save_path):
    # Clean previous log file if it exist
    os.remove(save_path)

print_every = args.print_every
printer = utils_rl.RLExamplePrinter(
    print_every,
    n_samples,
    save_path=save_path,
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

if args.fluency_min_max:
    fluency_model = FluencyRelativeScore()

    log_prob_min, log_prob_max = pre_process_min_max_fluency_log_probs(
        fluency_model=fluency_model, dataloader=train_dataloader
    )
    del fluency_model
else:
    log_prob_min, log_prob_max = None, None

reward_components_weights = args.reward_components_weights
if reward_components_weights:
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
        {
            "name": "fluency_lm",
            "model": FluencyRelativeScore(
                log_prob_min=log_prob_min, log_prob_max=log_prob_max
            ),
            "sign": 1,
        },
        {
            "name": "fluency_disc",
            "model": TextDiscriminator(retrain_every=800, fp16=True),
            "sign": 1,
        },
        {
            "name": "gr_repeat_penalty",
            "model": RepeatNGramPenalty(gram=3),
            "sign": -1,
            "weight": 2.0,
        },
        {
            "name": "gr_brevity_penalty",
            "model": RelativeBrevityPenalizer(min_ratio=0.6, max_ratio=1.3),
            "sign": -1,
            "weight": 2.0,
        },
        {
            "name": "gr_hallucination_penalty",
            "model": NERInaccuracyPenalty(),
            "sign": -1,
            "weight": 2.0,
        },
    ]
else:
    scorers = [
        {
            "name": "coverage",
            "model": coverage_model,
            "sign": 1,
            "weight": 1.0,
        },
        {
            "name": "simple_syn",
            "model": SimplicitySyntacticScore(),
            "sign": 1,
            "weight": 1.0,
        },
        {
            "name": "simple_lex",
            "model": SimplicityLexicalScore(target_shift=0.4, word_change_ratio=0.15),
            "sign": 1,
            "weight": 1.0,
        },
        {
            "name": "fluency_lm",
            "model": FluencyRelativeScore(
                log_prob_min=log_prob_min, log_prob_max=log_prob_max
            ),
            "sign": 1,
        },
        {
            "name": "fluency_disc",
            "model": TextDiscriminator(retrain_every=800, fp16=True),
            "weight": 1.0,
            "sign": 1,
        },
        {
            "name": "gr_repeat_penalty",
            "model": RepeatNGramPenalty(gram=3),
            "sign": -1,
            "weight": 1.0,
        },
        {
            "name": "gr_brevity_penalty",
            "model": RelativeBrevityPenalizer(min_ratio=0.6, max_ratio=1.3),
            "sign": -1,
            "weight": 1.0,
        },
        {
            "name": "gr_hallucination_penalty",
            "model": NERInaccuracyPenalty(),
            "sign": -1,
            "weight": 1.0,
        },
    ]


scorer = utils_scoring.ScorerWrapper(
    scorers, scoring_method=args.scoring, max_batch_size=12
)
T_start, T_last_best = time.time(), time.time()
temperature = 1.0

eval_frequency = 10

gene_params = {
    "max_output_length": max_seq_length,
    "sample": True,
    "num_runs": n_samples,
    "no_repeat_ngram": 5,
    "max_batch_size": 12,
    "no_copy_ngram": 7,
    "temperature": temperature,
}

if include_original:
    # We also increase by one the n_samples since we will add the original sentence
    batch_sample_size = n_samples + 1
else:
    batch_sample_size = n_samples

compute_eval_lexile = args.compute_eval_lexile

print("--- Doing evaluation of the model on the val set ---")
scores = evaluate_model(
    model=simplifier,
    coverage_model=coverage_model,
    dataloader=val_dataloader,
    lexile_compute=compute_eval_lexile,
)

eval_log = {f"val/{k}": v for k, v in scores.items()}
eval_log.update({"training_step": 0})
wandb.log(eval_log)

for idx, paragraphs in enumerate(train_dataloader):
    idx += 1
    T_batch_start = time.time()

    # Doing real, sampled generation
    gens_out = simplifier.generate(paragraphs, **gene_params)

    if include_original:
        # We also include the original in the generated as a ground truth
        # in an attempt to alleviate catastrophic failure.
        # To do so, we add the sentences as the output text and add the output tokens.
        gens_out[0].append(
            {
                "output_text": paragraphs[0],
                "output_tokens": simplifier.tokenizer.encode(
                    paragraphs[0], add_special_tokens=False
                )[: (simplifier.max_output_length - 1)],
            }
        )

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

    batch_RS_j = RS_j.reshape(train_batch_size, batch_sample_size)

    R_Overline_S = batch_RS_j.mean(dim=1)

    timer.tick("all_scores")
    # The first loss term is the R Overline S minus RS_j (see equation 5 in article).
    first_loss_term = R_Overline_S - batch_RS_j
    n_diff_pos, n_diff_neg = (first_loss_term < -0.02).long().sum().item(), (
        first_loss_term > 0.02
    ).long().sum().item()
    # We also increase by one the n_samples since we added the original sentence
    print(
        "[%d steps out of %d] [%d samples] %d above avg and %d below avg with a 0.02 margin."
        % (
            idx,
            max_steps,
            train_batch_size * batch_sample_size,
            n_diff_pos,
            n_diff_neg,
        )
    )

    diversity = len(set(generateds)) / len(generateds)
    temperature = thermostat.log_diversity(diversity)
    loss = rl_crit(unlooped_paragraphs, generateds_tokenized, first_loss_term)
    timer.tick("optim")

    batch_time = time.time() - T_batch_start
    log_obj = {
        "training_step": idx,
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

    # Run the Checkpoint engine
    current_score = np.mean(scorer_returns["total_scores"])
    is_best = ckpter.tick(current_score)
    if is_best:  # Run the inspection dataset through
        T_last_best = time.time()

    # Run the Printing engine
    printer.tick(
        paragraphs, generateds, scorer_returns, include_original=include_original
    )

    # Since each Wandb.log increase the step, we log the training with the eval to better align results
    if (idx % eval_frequency) == 0 or idx == max_steps:
        torch.cuda.empty_cache()
        print("--- Doing evaluation of the model on the val set ---")
        scores = evaluate_model(
            model=simplifier,
            coverage_model=coverage_model,
            dataloader=val_dataloader,
            lexile_compute=compute_eval_lexile,
        )

        log_obj.update({f"val/{k}": v for k, v in scores.items()})

    wandb.log(log_obj)

    if idx == 100:
        # The first 100 steps, we evaluate it each 10 steps
        # Then, for the steps between 100 and 1000, we evaluate it each 100 steps
        # Thus, we raise the eval_frequency
        eval_frequency = 100
        print_every = 300
    elif idx == 1000:
        # Then, for the steps between 1000 and 10 000, we evaluate it each 1000 steps
        eval_frequency = 1000
    elif idx == 10000:
        # Then, for the steps between 10 000 and max_steps, we evaluate it each 10 000 steps
        eval_frequency = 10000

print("--- Doing evaluation of the model on the test set ---")
scores = evaluate_model(
    model=simplifier,
    coverage_model=coverage_model,
    dataloader=test_dataloader,
    lexile_compute=True,
)

test_log_obj = {f"test/{k}": v for k, v in scores.items()}
test_log_obj.update({"training_step": max_steps})
wandb.log(test_log_obj)
