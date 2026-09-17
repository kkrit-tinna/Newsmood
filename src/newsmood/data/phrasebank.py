"""Financial PhraseBank: load, verify config nesting, and build splits.

`datasets.load_dataset("takala/financial_phrasebank", ...)` no longer works:
the repo only ships a legacy loading script, and `datasets>=4.0` dropped
script-based loading. We fetch the same raw zip the script used to fetch
and parse it with its exact logic instead. See IMPLEMENTATION_GUIDE.md §9.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import datasets
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download

from newsmood.config import Settings, get_settings

LABEL_NAMES = ["negative", "neutral", "positive"]

_ZIP_PATH_IN_REPO = "data/FinancialPhraseBank-v1.0.zip"

# Pinned against the zip fetched 2026-09-17. Mirrors the model-revision
# pinning pattern in CLAUDE.md — the raw file isn't versioned upstream
# (no Parquet conversion, no release tags), so this is the only thing
# stopping an upstream content change from silently reaching the parser.
# Re-pin deliberately (after diffing old vs. new content) if this legitimately
# needs to change; don't just update it to make a failure go away.
_ZIP_SHA256 = "0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6"

_AGREE_TO_FILENAME = {
    "sentences_allagree": "Sentences_AllAgree.txt",
    "sentences_75agree": "Sentences_75Agree.txt",
    "sentences_66agree": "Sentences_66Agree.txt",
    "sentences_50agree": "Sentences_50Agree.txt",
}


class PhraseBankSourceChanged(RuntimeError):
    """The downloaded PhraseBank zip no longer matches what this module was written against."""


def _raw_lines(repo_id: str, config_name: str) -> list[str]:
    zip_path = hf_hub_download(repo_id, _ZIP_PATH_IN_REPO, repo_type="dataset")

    actual_sha256 = hashlib.sha256(Path(zip_path).read_bytes()).hexdigest()
    if actual_sha256 != _ZIP_SHA256:
        raise PhraseBankSourceChanged(
            f"{repo_id}/{_ZIP_PATH_IN_REPO} sha256 is {actual_sha256}, expected {_ZIP_SHA256}. "
            "The upstream file has changed since this loader was written against it — "
            "verify the new content by hand (row counts, file names, line format) before "
            "re-pinning _ZIP_SHA256, rather than just updating the constant."
        )

    filename = _AGREE_TO_FILENAME[config_name]
    member = f"FinancialPhraseBank-v1.0/{filename}"
    with zipfile.ZipFile(zip_path) as z:
        try:
            raw = z.read(member).decode("iso-8859-1")
        except KeyError:
            raise PhraseBankSourceChanged(
                f"{member} not found in {_ZIP_PATH_IN_REPO} (checksum matched, so this "
                "shouldn't happen — the zip layout assumed by _AGREE_TO_FILENAME may be stale)."
            ) from None
    return raw.splitlines()


def load_config(repo_id: str, config_name: str) -> Dataset:
    """Load one agreement config as a Dataset with ClassLabel `label`."""
    sentences, labels = [], []
    for lineno, line in enumerate(_raw_lines(repo_id, config_name), start=1):
        if not line.strip():
            continue
        if "@" not in line:
            raise PhraseBankSourceChanged(
                f"{config_name} line {lineno} has no '@' separator: {line[:80]!r}. "
                "Expected 'sentence@label' — the source line format may have changed."
            )
        sentence, label = line.rsplit("@", 1)
        sentences.append(sentence.strip())
        labels.append(label.strip())
    features = datasets.Features(
        {
            "sentence": datasets.Value("string"),
            "label": datasets.features.ClassLabel(names=LABEL_NAMES),
        }
    )
    return Dataset.from_dict({"sentence": sentences, "label": labels}, features=features)


@dataclass
class NestingResult:
    row_counts: dict[str, int]
    violations: dict[str, list[str]]

    @property
    def ok(self) -> bool:
        return all(len(v) == 0 for v in self.violations.values())


def nesting_report(repo_id: str, configs: list[str]) -> NestingResult:
    """Verify configs[i]'s sentences are a subset of configs[i+1]'s (least to most agreement).

    `configs` must be ordered smallest to largest (allagree, 75agree, 66agree, 50agree).
    Checks set containment on sentence text, not just row counts — matching counts
    do not imply matching sentences.
    """
    sentence_sets = {}
    row_counts = {}
    for name in configs:
        ds = load_config(repo_id, name)
        sentence_sets[name] = set(ds["sentence"])
        row_counts[name] = len(ds)

    violations: dict[str, list[str]] = {}
    for smaller, larger in zip(configs, configs[1:]):
        missing = sentence_sets[smaller] - sentence_sets[larger]
        violations[f"{smaller} ⊄ {larger}"] = sorted(missing)

    return NestingResult(row_counts=row_counts, violations=violations)


def make_splits(dataset: Dataset, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> DatasetDict:
    """Stratified 3-way split. Both peels are stratified — stratifying only
    the first split lets train/val drift from the true label distribution."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    split1 = dataset.train_test_split(test_size=test_ratio, seed=seed, stratify_by_column="label")
    train_val, test = split1["train"], split1["test"]

    val_of_remainder = val_ratio / (train_ratio + val_ratio)
    split2 = train_val.train_test_split(test_size=val_of_remainder, seed=seed, stratify_by_column="label")
    train, val = split2["train"], split2["test"]

    return DatasetDict({"train": train, "val": val, "test": test})


def label_distribution(dataset: Dataset) -> dict[str, dict[str, float | int]]:
    names = dataset.features["label"].names
    counts = Counter(dataset["label"])
    total = len(dataset)
    dist = {names[i]: {"count": counts.get(i, 0), "share": counts.get(i, 0) / total} for i in range(len(names))}
    dist["majority_class_rate"] = max(v["share"] for v in dist.values() if isinstance(v, dict))
    return dist


def print_label_distribution(name: str, dataset: Dataset) -> None:
    dist = label_distribution(dataset)
    parts = [f"{k}={v['count']} ({v['share']:.1%})" for k, v in dist.items() if isinstance(v, dict)]
    print(f"  {name}: " + ", ".join(parts))


def token_length_stats(dataset: Dataset, tokenizer_name: str) -> dict[str, float]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    lengths = sorted(len(ids) for ids in tokenizer(list(dataset["sentence"]))["input_ids"])
    n = len(lengths)
    return {
        "mean": sum(lengths) / n,
        "p95": lengths[int(n * 0.95)],
        "max": lengths[-1],
    }


def _sentence_set_hash(sentences: list[str]) -> str:
    joined = "\n".join(sorted(sentences))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def save_splits(splits: DatasetDict, out_dir: Path, settings: Settings) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_json(str(out_dir / f"{name}.jsonl"))

    meta = {
        "phrasebank_config": settings.dataset.phrasebank_config,
        "train_ratio": settings.dataset.train_ratio,
        "val_ratio": settings.dataset.val_ratio,
        "test_ratio": settings.dataset.test_ratio,
        "seed": settings.dataset.seed,
        "row_counts": {name: len(split) for name, split in splits.items()},
        "datasets_version": datasets.__version__,
        "test_sentence_sha256": _sentence_set_hash(splits["test"]["sentence"]),
    }
    with open(out_dir / "splits_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_splits(out_dir: Path) -> DatasetDict:
    """Load persisted splits. Fails loudly if missing — never silently
    regenerates, since a `datasets` version bump could produce a different
    split under the same seed and invalidate every number measured before it."""
    meta_path = out_dir / "splits_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No splits_meta.json in {out_dir}. Run "
            "`python -m newsmood.data.phrasebank --build-splits` first — "
            "splits are never regenerated implicitly."
        )
    label_feature = datasets.features.ClassLabel(names=LABEL_NAMES)
    splits = DatasetDict(
        {
            name: Dataset.from_json(str(out_dir / f"{name}.jsonl")).cast_column("label", label_feature)
            for name in ("train", "val", "test")
        }
    )
    return splits


def verify_splits(out_dir: Path) -> None:
    """Recompute the persisted test-split hash and raise on mismatch."""
    meta_path = out_dir / "splits_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    test = Dataset.from_json(str(out_dir / "test.jsonl"))
    actual_hash = _sentence_set_hash(test["sentence"])
    if actual_hash != meta["test_sentence_sha256"]:
        raise ValueError(
            f"Split verification failed for {out_dir}: test-set hash "
            f"{actual_hash} does not match recorded {meta['test_sentence_sha256']}. "
            "The persisted splits have changed since they were written — "
            "do not trust numbers measured against them."
        )


def write_sample(test_split: Dataset, path: Path, n: int, seed: int) -> None:
    sample = test_split.shuffle(seed=seed).select(range(n))
    label_names = sample.features["label"].names
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in sample:
            f.write(json.dumps({"sentence": row["sentence"], "label": label_names[row["label"]]}) + "\n")


def _build_splits(settings: Settings) -> DatasetDict:
    dataset = load_config(settings.dataset.repo_id, settings.dataset.phrasebank_config)
    splits = make_splits(
        dataset,
        settings.dataset.train_ratio,
        settings.dataset.val_ratio,
        settings.dataset.test_ratio,
        settings.dataset.seed,
    )
    save_splits(splits, Path(settings.dataset.splits_dir), settings)
    write_sample(
        splits["test"],
        Path(settings.dataset.sample_path),
        settings.dataset.sample_size,
        settings.dataset.seed,
    )
    return splits


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="print split sizes and majority-class rate")
    parser.add_argument("--build-splits", action="store_true", help="build and persist splits + sample file")
    parser.add_argument("--check-nesting", action="store_true", help="verify the four configs nest as claimed")
    args = parser.parse_args()

    settings = get_settings()
    splits_dir = Path(settings.dataset.splits_dir)

    if args.check_nesting:
        result = nesting_report(settings.dataset.repo_id, settings.dataset.agreement_configs)
        print("Row counts:", result.row_counts)
        if result.ok:
            print("Nesting holds: allagree ⊆ 75agree ⊆ 66agree ⊆ 50agree")
        else:
            for pair, missing in result.violations.items():
                if missing:
                    print(f"VIOLATION {pair}: {len(missing)} sentence(s) not contained")
            raise SystemExit(1)

    if args.build_splits:
        splits = _build_splits(settings)
        print("Splits built:")
        for name, split in splits.items():
            print_label_distribution(name, split)

    if args.summary:
        splits = load_splits(splits_dir)
        verify_splits(splits_dir)
        dist = label_distribution(splits["test"])
        print(f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
        print(f"majority_class_rate={dist['majority_class_rate']:.4f}")


if __name__ == "__main__":
    _main()
