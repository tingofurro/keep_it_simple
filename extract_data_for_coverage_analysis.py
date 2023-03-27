import os.path

import pandas as pd
import wandb

api = wandb.Api(timeout=120)

runs = api.runs("davebulaval/keep_it_simple")

for run in runs:
    if "coverage" in run.name and "archive" not in run.name:
        pd.DataFrame(
            [
                row
                for row in run.scan_history(
                    keys=[
                        "coverage_original_sentence",
                        "coverage_all_masked_words_in_sentence",
                        "coverage_effective_mask_ratio",
                        "mean_coverage_scores",
                    ]
                )
            ],
            columns=[
                "coverage_original_sentence",
                "coverage_all_masked_words_in_sentence",
                "coverage_effective_mask_ratio",
                "mean_coverage_scores",
            ],
        ).to_csv(
            os.path.join(
                "results", f"{run.name}_run_{run.name}_components_steps_values.csv"
            ),
            index=False,
        )
