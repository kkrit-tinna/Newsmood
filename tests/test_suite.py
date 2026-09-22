from pathlib import Path

import datasets
from datasets import Dataset, DatasetDict

from newsmood.config import DatasetSettings, LogregSettings, Settings, VaderSettings
from newsmood.data.phrasebank import LABEL_NAMES, save_splits
from newsmood.evaluation.suite import run_reference_and_baselines

_AGREEMENT_CONFIGS = [
    "sentences_allagree",
    "sentences_75agree",
    "sentences_66agree",
    "sentences_50agree",
]

# Overlapping vocabulary so a handful of unigrams survive LogReg's min_df=2.
_TRAIN_SENTENCES = [
    "the company reported a sharp decline in quarterly profit",
    "shares plunged after the company posted a steep loss",
    "the company warned of falling revenue and rising costs",
    "analysts cut ratings after the company reported weak earnings",
    "the company will report quarterly earnings next week",
    "the company held its annual shareholder meeting today",
    "the company announced a new product line this quarter",
    "the company confirmed its previous guidance for the year",
    "the company reported strong quarterly profit and growth",
    "shares rallied after the company posted record earnings",
    "the company raised guidance after strong revenue growth",
    "analysts raised ratings after the company reported strong earnings",
]
_TRAIN_LABELS = [
    "negative",
    "negative",
    "negative",
    "negative",
    "neutral",
    "neutral",
    "neutral",
    "neutral",
    "positive",
    "positive",
    "positive",
    "positive",
]
_TEST_SENTENCES = [
    "the company posted a modest loss this quarter",
    "the company held steady with flat quarterly earnings",
    "the company reported strong growth in quarterly profit",
]
_TEST_LABELS = ["negative", "neutral", "positive"]


def _labeled_dataset(sentences: list[str], labels: list[str]) -> Dataset:
    features = datasets.Features(
        {
            "sentence": datasets.Value("string"),
            "label": datasets.features.ClassLabel(names=LABEL_NAMES),
        }
    )
    return Dataset.from_dict({"sentence": sentences, "label": labels}, features=features)


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
        vader=VaderSettings(positive_threshold=0.05, negative_threshold=-0.05),
        logreg=LogregSettings(
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            max_features=20000,
            class_weight="balanced",
            max_iter=1000,
            artifact_dir=str(tmp_path / "artifacts"),
        ),
    )


def _save_fixture_splits(settings: Settings) -> None:
    splits = DatasetDict(
        {
            "train": _labeled_dataset(_TRAIN_SENTENCES, _TRAIN_LABELS),
            "val": _labeled_dataset(_TRAIN_SENTENCES[:3], _TRAIN_LABELS[:3]),
            "test": _labeled_dataset(_TEST_SENTENCES, _TEST_LABELS),
        }
    )
    save_splits(splits, Path(settings.dataset.splits_dir), settings)


def test_run_reference_and_baselines_returns_one_row_per_method(tmp_path):
    settings = _settings(tmp_path)
    _save_fixture_splits(settings)

    table = run_reference_and_baselines(settings)

    assert set(table.keys()) == {"majority_class", "stratified_random", "vader", "logreg"}
    for row in table.values():
        assert set(row.keys()) == {"accuracy", "macro_f1", "weighted_f1", "per_class_recall", "confusion", "labels"}
        assert row["labels"] == LABEL_NAMES
