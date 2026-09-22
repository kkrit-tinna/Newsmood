from pathlib import Path

import datasets
from datasets import Dataset, DatasetDict

from newsmood.baselines.logreg import (
    fit,
    load_artifacts,
    load_vectorizer,
    predict,
    predict_test_split,
)
from newsmood.config import DatasetSettings, LogregSettings, Settings
from newsmood.data.phrasebank import LABEL_NAMES, save_splits
from newsmood.preprocessing import clean_for_tfidf

_AGREEMENT_CONFIGS = [
    "sentences_allagree",
    "sentences_75agree",
    "sentences_66agree",
    "sentences_50agree",
]

# Deliberately overlapping vocabulary across classes, so a handful of
# unigrams survive min_df=2 / max_df=0.95 on a corpus this small.
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


def _settings(tmp_path: Path, **logreg_overrides) -> Settings:
    logreg_kwargs = dict(
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        max_features=20000,
        class_weight="balanced",
        max_iter=1000,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    logreg_kwargs.update(logreg_overrides)
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
        logreg=LogregSettings(**logreg_kwargs),
    )


def _save_fixture_splits(
    settings: Settings,
    test_sentences: list[str] | None = None,
    test_labels: list[str] | None = None,
) -> None:
    """Write train/val/test straight to disk through save_splits(), bypassing
    make_splits()'s random assignment — these tests need to control exactly
    which sentences land in train vs. test."""
    splits = DatasetDict(
        {
            "train": _labeled_dataset(_TRAIN_SENTENCES, _TRAIN_LABELS),
            "val": _labeled_dataset(_TRAIN_SENTENCES[:3], _TRAIN_LABELS[:3]),
            "test": _labeled_dataset(
                test_sentences if test_sentences is not None else _TEST_SENTENCES,
                test_labels if test_labels is not None else _TEST_LABELS,
            ),
        }
    )
    save_splits(splits, Path(settings.dataset.splits_dir), settings)


# --- 1. leakage: a test-only token must never reach the fitted vocabulary ---


def test_fit_does_not_leak_test_only_tokens_into_vocabulary(tmp_path):
    settings = _settings(tmp_path)
    _save_fixture_splits(
        settings,
        # "zqxleak" appears in 2 test rows, 0 train rows — 2, not 1, so it
        # would survive min_df=2 if it ever reached the fit call. If this
        # regresses to fitting on train+test (or on test alone), the token
        # ends up in the vocabulary; fitting on train alone, it can't.
        test_sentences=[
            "zqxleak shares tumbled after the announcement",
            "investors flagged the zqxleak report as unusual",
            "the company reported strong quarterly profit and growth",
        ],
        test_labels=["negative", "neutral", "positive"],
    )

    predict_test_split(settings)

    vectorizer = load_vectorizer(settings)
    assert "zqxleak" not in vectorizer.vocabulary_


# --- 2. preprocessing: exactly the four operations, nothing else ---


def test_clean_for_tfidf_lowercases_collapses_whitespace_strips_urls_and_tickers():
    text = "  $AAPL Shares ROSE   after http://example.com/news and www.example.org/x  "
    assert clean_for_tfidf(text) == "shares rose after and"


def test_clean_for_tfidf_leaves_punctuation_alone():
    text = "Profit fell, sharply! (Again.)"
    assert clean_for_tfidf(text) == "profit fell, sharply! (again.)"


# --- 3. persistence: save -> load round-trips to identical predictions ---


def test_persisted_artifacts_round_trip_to_identical_predictions(tmp_path):
    settings = _settings(tmp_path)
    _save_fixture_splits(settings)

    sentences, y_true, y_pred = predict_test_split(settings)

    loaded_vectorizer, loaded_classifier = load_artifacts(settings)
    reloaded_pred = predict(loaded_vectorizer, loaded_classifier, sentences)

    assert reloaded_pred == y_pred


# --- 4. predictions never leave the three-label space ---


def test_predict_test_split_predictions_are_valid_labels(tmp_path):
    settings = _settings(tmp_path)
    _save_fixture_splits(settings)

    _, y_true, y_pred = predict_test_split(settings)

    assert len(y_pred) == len(y_true)
    assert set(y_pred) <= set(LABEL_NAMES)


def test_fit_predict_directly_also_stays_within_label_space(tmp_path):
    settings = _settings(tmp_path)
    vectorizer, classifier = fit(_TRAIN_SENTENCES, _TRAIN_LABELS, settings)

    y_pred = predict(vectorizer, classifier, _TEST_SENTENCES)

    assert len(y_pred) == len(_TEST_SENTENCES)
    assert set(y_pred) <= set(LABEL_NAMES)
