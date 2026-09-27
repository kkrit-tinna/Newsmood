"""ProsusAI/finbert, scored without further training: a reference row, not a
baseline to beat, and not a fair ceiling (IMPLEMENTATION_GUIDE.md T2.3, §9
2026-09-27).

FinBERT was fine-tuned on Financial PhraseBank itself, so it has very likely
seen many of our test sentences. Its number is contaminated, not zero-shot.

Its output order ({0: positive, 1: negative, 2: neutral} at the pinned
revision) differs from ours; models/classifier.py maps outputs by name.
"""

from __future__ import annotations

from pathlib import Path

from newsmood.config import Settings
from newsmood.data.phrasebank import LABEL_NAMES, load_splits, verify_splits
from newsmood.models.classifier import load_classifier, predict


def predict_test_split(settings: Settings) -> tuple[list[str], list[str], list[str]]:
    """Score the pinned test split with the pinned FinBERT checkpoint on CPU.

    CPU, not get_device(): this row is published in docs/baselines.md, and CPU
    inference is bit-reproducible where MPS is not.

    Returns (sentences, y_true, y_pred).
    """
    splits_dir = Path(settings.dataset.splits_dir)
    splits = load_splits(splits_dir)
    verify_splits(splits_dir)
    test = splits["test"]

    cfg = settings.finbert
    model, tokenizer = load_classifier(cfg.repo_id, revision=cfg.revision)

    sentences = list(test["sentence"])
    y_true = [LABEL_NAMES[i] for i in test["label"]]
    y_pred = predict(sentences, model, tokenizer, cfg.batch_size, "cpu")
    return sentences, y_true, y_pred
