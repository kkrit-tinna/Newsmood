import datasets
from datasets import Dataset

from newsmood.baselines.vader import label_for_compound, predict, predict_test_split
from newsmood.config import DatasetSettings, Settings, VaderSettings
from newsmood.data.phrasebank import LABEL_NAMES, make_splits, save_splits

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


# --- label_for_compound: pure threshold logic, no lexicon involved ---


def test_label_for_compound_positive_at_threshold():
    assert label_for_compound(0.05, POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD) == "positive"


def test_label_for_compound_negative_at_threshold():
    assert label_for_compound(-0.05, POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD) == "negative"


def test_label_for_compound_neutral_between_thresholds():
    assert label_for_compound(0.0, POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD) == "neutral"
    assert label_for_compound(0.049, POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD) == "neutral"
    assert label_for_compound(-0.049, POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD) == "neutral"


# --- predict: exercises the bundled vaderSentiment lexicon, no network ---


def test_predict_obvious_positive_and_negative_sentences():
    sentences = [
        "The company reported excellent, outstanding growth this quarter.",
        "The company reported a terrible, disastrous collapse in earnings.",
    ]
    labels = predict(sentences, POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD)
    assert labels == ["positive", "negative"]


def test_predict_returns_one_label_per_sentence():
    sentences = ["Stock prices held steady.", "Markets were quiet today.", "Shares closed flat."]
    labels = predict(sentences, POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD)
    assert len(labels) == len(sentences)
    assert all(label in LABEL_NAMES for label in labels)


# --- predict_test_split: wires config + persisted splits together ---


def _settings(splits_dir: str) -> Settings:
    return Settings(
        dataset=DatasetSettings(
            repo_id="takala/financial_phrasebank",
            phrasebank_config="sentences_75agree",
            agreement_configs=[
                "sentences_allagree",
                "sentences_75agree",
                "sentences_66agree",
                "sentences_50agree",
            ],
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            tokenizer="distilbert-base-uncased",
            splits_dir=splits_dir,
            sample_size=10,
            sample_path=str(splits_dir) + "/sample.jsonl",
        ),
        vader=VaderSettings(positive_threshold=POSITIVE_THRESHOLD, negative_threshold=NEGATIVE_THRESHOLD),
    )


def _skewed_dataset(n_neutral=60, n_positive=25, n_negative=15) -> Dataset:
    features = datasets.Features(
        {
            "sentence": datasets.Value("string"),
            "label": datasets.features.ClassLabel(names=LABEL_NAMES),
        }
    )
    sentences, labels = [], []
    for i in range(n_neutral):
        sentences.append(f"neutral sentence {i}")
        labels.append("neutral")
    for i in range(n_positive):
        sentences.append(f"positive sentence {i}")
        labels.append("positive")
    for i in range(n_negative):
        sentences.append(f"negative sentence {i}")
        labels.append("negative")
    return Dataset.from_dict({"sentence": sentences, "label": labels}, features=features)


def test_predict_test_split_returns_aligned_sentences_and_labels(tmp_path):
    ds = _skewed_dataset()
    splits = make_splits(ds, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    settings = _settings(str(tmp_path / "splits"))
    save_splits(splits, tmp_path / "splits", settings)

    sentences, y_true, y_pred = predict_test_split(settings)

    assert len(sentences) == len(y_true) == len(y_pred) == len(splits["test"])
    assert set(y_true) <= set(LABEL_NAMES)
    assert set(y_pred) <= set(LABEL_NAMES)
