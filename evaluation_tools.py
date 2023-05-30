from typing import List

import nltk
from evaluate import load
from textstat import textstat

sari = load("sari")
bleu = load("bleu")


class RatioMetric:
    def __init__(self, n: int) -> None:
        self.n = n
        self._running_score = 0  # Number of time the simplified metric score is lower than the original sentence

    def compute_ratio(self) -> float:
        """
        compute % of metric_simplified_lowered / n * 100
        """
        return round(self._running_score / self.n * 100, 4)


class FKGLRatio(RatioMetric):
    def __init__(self, n: int) -> None:
        super().__init__(n)
        self._fkgl_scores = []

    def compute_fkgl_scores(self, references: str, predictions: str) -> None:
        """
        Compute the FKGL score for the original sentence (references) and compute the FKGL score for the simplified
        sentence (predictions). If the simplified FKGL score is lower than the original FKGL score we increment a
        counter to compute the number of time simplified score is lower than the original sentence.

        Algorithm:

        compute FKGL of the original sentence -> fkgl_original
        compute FKGL of the simplified sentence -> fkgl_simplified
        if fkgl_simplified < fkgl_original: fkgl_simplified_lowered += 1
        """
        references_fkgl_score = textstat.flesch_kincaid_grade(references)
        predictions_fkgl_score = textstat.flesch_kincaid_grade(predictions)

        if predictions_fkgl_score < references_fkgl_score:
            self._running_score += 1

        self._fkgl_scores.append((references_fkgl_score, predictions_fkgl_score))


def compute_lexile(text: str) -> float:
    # todo add: compute %Lexile waiting feedback from email sent March 29th (72 hours delay from email).
    ## Use API https://partnerhelp.metametricsinc.com/concept/c_la_api_overview.html
    ## Need to develop code to process it
    ## Need an API key
    ## Possible implementation in Python https://github.com/altarac/Lexile-scoring
    return 1


class LexileRatio(RatioMetric):
    def __init__(self, n: int) -> None:
        super().__init__(n)
        self._lexile_scores = []

    def compute_lexile_scores(self, references: str, predictions: str) -> None:
        """
        Compute the Lexile score for the original sentence (references) and compute the Lexile score for the simplified
        sentence (predictions). If the simplified LExile score is lower than the original LExile score we increment a
        counter to compute the number of time simplified score is lower than the original sentence.

        Algorithm:

        compute Lexile of the original sentence -> lexile_original
        compute LExile of the simplified sentence -> lexile_simplified
        if lexile_simplified < lexile_original: lexile_simplified_lowered += 1

        compute % of lexile_simplified_lowered / 500 * 100
        """
        references_lexile_score = compute_lexile(references)
        predictions_lexile_score = compute_lexile(predictions)

        if predictions_lexile_score < references_lexile_score:
            self._running_score += 1

        self._lexile_scores.append((references_lexile_score, predictions_lexile_score))


def compute_sari(references: List, predictions: List) -> float:
    """
    About SARI:
    - https://huggingface.co/spaces/evaluate-metric/sari
    - 0 is the worst, 100 is the best.
    """
    sari_score = sari.compute(
        sources=references, predictions=predictions, references=[references]
    )
    return sari_score["sari"]


def compute_bleu(references: List, predictions: List) -> float:
    """
    About BLEU:
    - https://huggingface.co/spaces/evaluate-metric/bleu
    - 0 is the worst, 100 is the best.
    todo: SacreBLEU, Google BLEU or BLEU? To validate with author -> email sent March 29th 2023
    """
    try:
        bleu_score = bleu.compute(predictions=predictions, references=references)
        return round(bleu_score["bleu"] * 100, 4)
    except ZeroDivisionError:
        return 0.00


def compute_compression_rate(references: List, predictions: List) -> float:
    """
    About compression rate:
    - We compute it as 1 minus the number of words (tokenize using NLTK word tokenizer) of the simplified sentence over
        the number of words in the original sentence.
    - https://en.wikipedia.org/wiki/Data_compression_ratio
    - todo: validate with authors how they did it -> email sent March 29th 2023
    """
    number_of_words_references = len(nltk.tokenize.word_tokenize(references[0]))
    number_of_words_predictions = len(nltk.tokenize.word_tokenize(predictions[0]))

    return 1 - number_of_words_predictions / number_of_words_references
