from statistics import mean

import torch
from tqdm import tqdm

from evaluation_tools import (
    FKGLRatio,
    compute_sari,
    compute_bleu,
    compute_compression_rate,
    LexileRatio,
)
from model_generator import Generator
from model_salience import CoverageModel


def evaluate_model(
    model: Generator, coverage_model: CoverageModel, dataloader, n: int = 500
):
    with torch.no_grad():
        fkgl_ratio = FKGLRatio(n=n)
        lexile_ratio = LexileRatio(n=n)

        sari_scores = []
        bleu_scores = []
        compression_rates = []
        coverage_rates = []

        for idx, paragraphs in tqdm(enumerate(dataloader), total=n):
            if idx < n:
                gene_params = {
                    "max_output_length": 90,
                    "sample": True,
                    "num_runs": 8,
                    "no_repeat_ngram": 5,
                    "max_batch_size": 12,
                    "no_copy_ngram": 7,
                    "temperature": 1.0,
                }

                # We sort the prediction
                predictions = model.generate(
                    paragraphs, **gene_params, sort_score=True
                )[0]
                # We take the best prediction to be evaluated
                best_predictions = predictions[0]

                # paragraphs: List
                # best_predictions: Dict
                # bes_predictions["output_text"]: str

                sari_score = compute_sari(
                    references=paragraphs, predictions=[best_predictions["output_text"]]
                )
                sari_scores.append(sari_score)

                bleu_score = compute_bleu(
                    references=paragraphs, predictions=[best_predictions["output_text"]]
                )
                bleu_scores.append(bleu_score)

                fkgl_ratio.compute_fkgl_scores(
                    references=paragraphs[0],
                    predictions=best_predictions["output_text"],
                )

                compression_rate = compute_compression_rate(
                    references=paragraphs, predictions=[best_predictions["output_text"]]
                )
                compression_rates.append(compression_rate)

                # About Coverage:
                # 0 is the worst, 100 is the best.
                coverage_scores = coverage_model.score(
                    bodies=paragraphs, decodeds=[best_predictions["output_text"]]
                )
                coverage_rates.append(round(coverage_scores["scores"][0] * 100, 4))

                lexile_ratio.compute_lexile_scores(
                    references=paragraphs[0],
                    predictions=best_predictions["output_text"],
                )
            else:
                break
        average_sari_score = mean(sari_scores)
        average_bleu_score = mean(bleu_scores)
        fkgl_ratio_score = fkgl_ratio.compute_ratio()
        compression_rate_score = mean(compression_rates)
        coverage_rate_score = mean(coverage_rates)

        return {
            "average_sari_score": average_sari_score,
            "average_bleu_score": average_bleu_score,
            "fkgl_ratio_score": fkgl_ratio_score,
            "compression_rate_score": compression_rate_score,
            "coverage_rate_score": coverage_rate_score,
        }
