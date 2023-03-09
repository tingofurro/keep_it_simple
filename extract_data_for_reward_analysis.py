import os.path

import pandas as pd
import wandb

api = wandb.Api(timeout=120)

runs = api.runs("davebulaval/keep_it_simple")

for run in runs:
    if "reward_analysis" in run.name:
        pd.DataFrame(
            [
                row
                for row in run.scan_history(
                    keys=[
                        "coverage_scores",
                        "simple_syn_scores",
                        "simple_lex_scores",
                        "fluency_lm_scores",
                        "fluency_disc_scores",
                        "total_scores",
                    ]
                )
            ]
        ).to_csv(
            os.path.join(
                "results", f"{run.name}_run_rewards_components_steps_values.csv"
            ),
            index=False,
        )

        pd.DataFrame(
            [
                row
                for row in run.scan_history(
                    keys=[
                        "gr_repeat_scores",
                        "gr_brevity_scores",
                        "gr_hallucination_scores",
                    ]
                )
            ]
        ).to_csv(
            os.path.join(
                "results",
                f"{run.name}_run_rewards_components_steps_values_guardrails.csv",
            ),
            index=False,
        )
