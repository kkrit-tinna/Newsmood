"""VADER baseline: lexicon compound score thresholded into positive/neutral/negative.

Raw sentences go in unmodified — VADER, like the DistilBERT model, is not fed
through `preprocessing.py`. That module exists for TF-IDF only (CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from newsmood.config import Settings
from newsmood.data.phrasebank import LABEL_NAMES, load_splits


def label_for_compound(compound: float, positive_threshold: float, negative_threshold: float) -> str:
    if compound >= positive_threshold:
        return "positive"
    if compound <= negative_threshold:
        return "negative"
    return "neutral"


def predict(
    sentences: list[str],
    positive_threshold: float,
    negative_threshold: float,
) -> list[str]:
    analyzer = SentimentIntensityAnalyzer()
    predictions = []
    for sentence in sentences:
        compound = analyzer.polarity_scores(sentence)["compound"]
        predictions.append(label_for_compound(compound, positive_threshold, negative_threshold))
    return predictions


def predict_test_split(settings: Settings) -> tuple[list[str], list[str], list[str]]:
    """Run VADER over the pinned test split.

    Returns (sentences, y_true, y_pred) so callers can both score and
    inspect individual misclassifications.
    """
    splits_dir = Path(settings.dataset.splits_dir)
    splits = load_splits(splits_dir)
    test: Dataset = splits["test"]

    sentences = list(test["sentence"])
    y_true = [LABEL_NAMES[i] for i in test["label"]]
    y_pred = predict(
        sentences,
        settings.vader.positive_threshold,
        settings.vader.negative_threshold,
    )
    return sentences, y_true, y_pred
