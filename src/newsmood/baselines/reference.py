"""Reference floors: majority-class and stratified-random.

Neither reads the sentence text — both predict from the training label
distribution alone. They exist so later numbers ("DistilBERT hits 82%")
have something to be measured against (IMPLEMENTATION_GUIDE.md T1.5).
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from datasets import Dataset

from newsmood.config import Settings
from newsmood.data.phrasebank import LABEL_NAMES, load_splits


def majority_label(labels: list[str]) -> str:
    if not labels:
        raise ValueError("labels is empty")
    return Counter(labels).most_common(1)[0][0]


def majority_class_predict(labels_train: list[str], n: int) -> list[str]:
    label = majority_label(labels_train)
    return [label] * n


def stratified_random_predict(labels_train: list[str], n: int, seed: int) -> list[str]:
    """Sample `n` predictions from the training label distribution."""
    counts = Counter(labels_train)
    population = list(counts.keys())
    weights = list(counts.values())
    return random.Random(seed).choices(population, weights=weights, k=n)


def _train_and_test_labels(settings: Settings) -> tuple[list[str], list[str], list[str]]:
    splits_dir = Path(settings.dataset.splits_dir)
    splits = load_splits(splits_dir)
    train: Dataset = splits["train"]
    test: Dataset = splits["test"]
    labels_train = [LABEL_NAMES[i] for i in train["label"]]
    sentences = list(test["sentence"])
    y_true = [LABEL_NAMES[i] for i in test["label"]]
    return labels_train, sentences, y_true


def majority_class_predict_test_split(settings: Settings) -> tuple[list[str], list[str], list[str]]:
    labels_train, sentences, y_true = _train_and_test_labels(settings)
    y_pred = majority_class_predict(labels_train, len(y_true))
    return sentences, y_true, y_pred


def stratified_random_predict_test_split(settings: Settings) -> tuple[list[str], list[str], list[str]]:
    labels_train, sentences, y_true = _train_and_test_labels(settings)
    y_pred = stratified_random_predict(labels_train, len(y_true), settings.dataset.seed)
    return sentences, y_true, y_pred
