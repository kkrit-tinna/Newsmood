import json

import datasets
import pytest
from datasets import Dataset

from newsmood.data.phrasebank import (
    LABEL_NAMES,
    NestingResult,
    label_distribution,
    load_splits,
    make_splits,
    nesting_report,
    save_splits,
    verify_splits,
    write_sample,
)
from newsmood.config import DatasetSettings, Settings


def _labeled_dataset(sentences: list[str], labels: list[str]) -> Dataset:
    features = datasets.Features(
        {
            "sentence": datasets.Value("string"),
            "label": datasets.features.ClassLabel(names=LABEL_NAMES),
        }
    )
    return Dataset.from_dict({"sentence": sentences, "label": labels}, features=features)


def _skewed_dataset(n_neutral=60, n_positive=25, n_negative=15) -> Dataset:
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
    return _labeled_dataset(sentences, labels)


def _settings(splits_dir: str, sample_path: str) -> Settings:
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
            sample_path=sample_path,
        )
    )


# --- label_distribution ---


def test_label_distribution_counts_and_majority_rate():
    ds = _skewed_dataset(n_neutral=60, n_positive=25, n_negative=15)
    dist = label_distribution(ds)
    assert dist["neutral"]["count"] == 60
    assert dist["positive"]["count"] == 25
    assert dist["negative"]["count"] == 15
    assert dist["neutral"]["share"] == pytest.approx(0.60)
    assert dist["majority_class_rate"] == pytest.approx(0.60)


# --- make_splits: stratification on both peels ---


def test_make_splits_sizes_and_disjoint():
    ds = _skewed_dataset(n_neutral=600, n_positive=250, n_negative=150)
    splits = make_splits(ds, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(ds)

    train_sentences = set(splits["train"]["sentence"])
    val_sentences = set(splits["val"]["sentence"])
    test_sentences = set(splits["test"]["sentence"])
    assert train_sentences.isdisjoint(val_sentences)
    assert train_sentences.isdisjoint(test_sentences)
    assert val_sentences.isdisjoint(test_sentences)


def test_make_splits_stratifies_train_and_val_not_just_test():
    ds = _skewed_dataset(n_neutral=600, n_positive=250, n_negative=150)
    splits = make_splits(ds, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    overall_rate = label_distribution(ds)["negative"]["share"]
    for name in ("train", "val", "test"):
        rate = label_distribution(splits[name])["negative"]["share"]
        assert rate == pytest.approx(overall_rate, abs=0.02), (
            f"{name} split's negative share {rate} drifted from overall {overall_rate}"
        )


# --- save/load/verify: pinning ---


def test_load_splits_fails_loudly_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_splits(tmp_path / "nonexistent")


def test_save_load_round_trip_preserves_labels(tmp_path):
    ds = _skewed_dataset(n_neutral=60, n_positive=25, n_negative=15)
    splits = make_splits(ds, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    settings = _settings(str(tmp_path / "splits"), str(tmp_path / "sample.jsonl"))

    save_splits(splits, tmp_path / "splits", settings)
    reloaded = load_splits(tmp_path / "splits")

    assert reloaded["train"].features["label"].names == LABEL_NAMES
    assert set(reloaded["test"]["sentence"]) == set(splits["test"]["sentence"])


def test_verify_splits_raises_on_tampered_data(tmp_path):
    ds = _skewed_dataset(n_neutral=60, n_positive=25, n_negative=15)
    splits = make_splits(ds, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    settings = _settings(str(tmp_path / "splits"), str(tmp_path / "sample.jsonl"))
    out_dir = tmp_path / "splits"
    save_splits(splits, out_dir, settings)

    verify_splits(out_dir)  # passes untouched

    lines = (out_dir / "test.jsonl").read_text().splitlines()
    row = json.loads(lines[0])
    row["sentence"] = "TAMPERED"
    lines[0] = json.dumps(row)
    (out_dir / "test.jsonl").write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError):
        verify_splits(out_dir)


def test_splits_meta_records_pinning_fields(tmp_path):
    ds = _skewed_dataset(n_neutral=60, n_positive=25, n_negative=15)
    splits = make_splits(ds, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    settings = _settings(str(tmp_path / "splits"), str(tmp_path / "sample.jsonl"))
    out_dir = tmp_path / "splits"
    save_splits(splits, out_dir, settings)

    meta = json.loads((out_dir / "splits_meta.json").read_text())
    for key in (
        "phrasebank_config",
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "seed",
        "row_counts",
        "datasets_version",
        "test_sentence_sha256",
    ):
        assert key in meta


# --- nesting_report: set containment, not just counts ---


def test_nesting_report_passes_when_truly_nested(monkeypatch):
    small = _labeled_dataset(["a", "b"], ["negative", "neutral"])
    large = _labeled_dataset(["a", "b", "c"], ["negative", "neutral", "positive"])

    def fake_load_config(repo_id, config_name):
        return {"small": small, "large": large}[config_name]

    monkeypatch.setattr("newsmood.data.phrasebank.load_config", fake_load_config)
    result = nesting_report("fake-repo", ["small", "large"])
    assert result.ok
    assert result.row_counts == {"small": 2, "large": 3}


def test_nesting_report_flags_same_size_but_different_sentences(monkeypatch):
    # Same row count on both sides, but not a subset relationship —
    # this is exactly the case row-count checks alone would miss.
    small = _labeled_dataset(["a", "b"], ["negative", "neutral"])
    large = _labeled_dataset(["a", "x"], ["negative", "positive"])

    def fake_load_config(repo_id, config_name):
        return {"small": small, "large": large}[config_name]

    monkeypatch.setattr("newsmood.data.phrasebank.load_config", fake_load_config)
    result = nesting_report("fake-repo", ["small", "large"])
    assert not result.ok
    assert result.violations["small ⊄ large"] == ["b"]


# --- write_sample ---


def test_write_sample_writes_n_rows_with_label_names(tmp_path):
    ds = _skewed_dataset(n_neutral=60, n_positive=25, n_negative=15)
    out_path = tmp_path / "sample.jsonl"
    write_sample(ds, out_path, n=10, seed=42)

    lines = out_path.read_text().splitlines()
    assert len(lines) == 10
    for line in lines:
        row = json.loads(line)
        assert row["label"] in LABEL_NAMES
        assert isinstance(row["sentence"], str)
