from argparse import Namespace
from typing import Dict, Union


def create_comparison_tag(args: Union[Namespace, Dict]):
    """
    Function to handle tag for Wandb visualisation.
    """
    if isinstance(args, Namespace):
        config = vars(args)
    else:
        config = args

    comparison_tag = "Baseline"
    if config.get("include_original"):
        comparison_tag = "Original in S"

    if not config.get("reward_components_weights"):
        # If True = include weights, if False = does not include weights.
        if comparison_tag == "Baseline":
            comparison_tag = "No weights"
        else:
            comparison_tag = f"{comparison_tag}, no weights"

    if config.get("fluency_min_max"):
        if comparison_tag == "Baseline":
            comparison_tag = "Fluency min-max"
        else:
            comparison_tag = f"{comparison_tag}, fluency min-max"

    return comparison_tag