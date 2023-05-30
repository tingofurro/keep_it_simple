import argparse
import os

import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader, RandomSampler

import utils_misc
from evaluation import evaluate_model
from model_generator import Generator
from model_salience import CoverageModel
from tools import bool_parse
from utils_dataset import cc_newsela_collate

freer_gpu = utils_misc.select_freer_gpu()

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_file",
    type=str,
    help="Use for example `gpt2_med_keep_it_simple.bin` provided in the codebase.",
)
parser.add_argument(
    "coverage_model_path",
    type=str,
    help="Coverage model bin file path.",
)
parser.add_argument(
    "--model_card",
    type=str,
    default="gpt2-medium",
    help="Either `gpt2` or `gpt2-medium`",
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=90,
    help="Maximum output length. Saves time if the sequences are short.",
)
parser.add_argument(
    "--saliency_min_max_normalization",
    type=bool_parse,
    default=False,
    help="Either or not to do min max normalization on the saliency scores.",
)


args = parser.parse_args()

coverage_model = CoverageModel(
    "nostop",
    model_file=args.coverage_model_path,
    fp16=True,
    is_soft=True,
    min_max_normalization=args.saliency_min_max_normalization,
)

model = Generator(args.model_card, max_output_length=args.max_seq_length)
model.reload(args.model_file)
model.eval()

dataset_df = pd.read_csv(os.path.join("datastore", "newsela_paired_0.2.csv"))

test_dataset_df = dataset_df[dataset_df["cut"] == "dev"]
test_dataset = Dataset.from_pandas(test_dataset_df)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    sampler=RandomSampler(test_dataset),
    drop_last=True,
    collate_fn=cc_newsela_collate,
)

scores = evaluate_model(
    model=model, coverage_model=coverage_model, dataloader=test_dataloader, n=500
)

print(scores)
