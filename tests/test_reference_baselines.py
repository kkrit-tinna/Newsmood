from pathlib import Path

import datasets
import pytest
from datasets import Dataset, DatasetDict

from newsmood.baselines.reference import (
    majority_class_predict,
    majority_class_predict_test_split,
    majority_label,
    stratified_random_predict,
    stratified_random_predict_test_split,
)
from newsmood.config import DatasetSettings, Settings
from newsmood.data.phrasebank import LABEL_NAMES, save_splits

_AGREEMENT_CONFIGS = [
    "sentences_allagree",
    "sentences_75agree",
    "sentences_66agree",
    "sentences_50agree",
]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        dataset=DatasetSettings(
            repo_id="takala/financial_phrasebank",
            phrasebank_config="sentences_75agree",
            agreement_configs=_AGREEMENT_CONFIGS,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            tokenizer="distilbert-base-uncased",
            splits_dir=str(tmp_path / "splits"),
            sample_size=3,
            sample_path=str(tmp_path / "sample.jsonl"),
        ),
    )


def _labeled_dataset(sentences: list[str], labels: list[str]) -> Dataset:
    features = datasets.Features(
        {
            "sentence": datasets.Value("string"),
            "label": datasets.features.ClassLabel(names=LABEL_NAMES),
        }
    )
    return Dataset.from_dict({"sentence": sentences, "label": labels}, features=features)


# --- majority_label / majority_class_predict: pure logic ---


def test_majority_label_picks_the_most_common_class():
    labels = ["neutral"] * 6 + ["positive"] * 3 + ["negative"] * 1
    assert majority_label(labels) == "neutral"


def test_majority_label_empty_raises():
    with pytest.raises(ValueError):
        majority_label([])


def test_majority_class_predict_always_returns_the_same_label():
    labels_train = ["neutral"] * 6 + ["positive"] * 3 + ["negative"] * 1
    predictions = majority_class_predict(labels_train, n=5)
    assert predictions == ["neutral"] * 5


# --- stratified_random_predict: pure logic ---


def test_stratified_random_predict_returns_n_predictions_within_label_space():
    labels_train = ["neutral"] * 6 + ["positive"] * 3 + ["negative"] * 1
    predictions = stratified_random_predict(labels_train, n=50, seed=42)
    assert len(predictions) == 50
    assert set(predictions) <= set(labels_train)


def test_stratified_random_predict_is_deterministic_for_a_fixed_seed():
    labels_train = ["neutral"] * 6 + ["positive"] * 3 + ["negative"] * 1
    first = stratified_random_predict(labels_train, n=50, seed=42)
    second = stratified_random_predict(labels_train, n=50, seed=42)
    assert first == second


def test_stratified_random_predict_never_draws_a_class_absent_from_training():
    labels_train = ["neutral"] * 10 + ["positive"] * 10  # no negative examples
    predictions = stratified_random_predict(labels_train, n=200, seed=42)
    assert "negative" not in predictions


# --- *_test_split: wires config + persisted splits together ---


def _save_fixture_splits(settings: Settings) -> None:
    train_sentences = [f"train sentence {i}" for i in range(20)]
    train_labels = ["neutral"] * 12 + ["positive"] * 5 + ["negative"] * 3
    test_sentences = [f"test sentence {i}" for i in range(10)]
    test_labels = ["neutral"] * 6 + ["positive"] * 3 + ["negative"] * 1

    splits = DatasetDict(
        {
            "train": _labeled_dataset(train_sentences, train_labels),
            "val": _labeled_dataset(train_sentences[:3], train_labels[:3]),
            "test": _labeled_dataset(test_sentences, test_labels),
        }
    )
    save_splits(splits, Path(settings.dataset.splits_dir), settings)


def test_majority_class_predict_test_split_matches_train_majority(tmp_path):
    settings = _settings(tmp_path)
    _save_fixture_splits(settings)

    sentences, y_true, y_pred = majority_class_predict_test_split(settings)

    assert len(sentences) == len(y_true) == len(y_pred) == 10
    assert set(y_pred) == {"neutral"}


def test_stratified_random_predict_test_split_stays_within_label_space(tmp_path):
    settings = _settings(tmp_path)
    _save_fixture_splits(settings)

    sentences, y_true, y_pred = stratified_random_predict_test_split(settings)

    assert len(sentences) == len(y_true) == len(y_pred) == 10
    assert set(y_pred) <= set(LABEL_NAMES)
