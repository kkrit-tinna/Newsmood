"""The fine-tuned DistilBERT row of the eval table (IMPLEMENTATION_GUIDE.md T2.3).

Loads the local `save_pretrained` artifact from T2.2 (`training.output_dir`),
not the Hub — T2.4 publishes it. Scored on CPU, like FinBERT, because the row
is published and CPU inference is bit-reproducible.
"""

from __future__ import annotations

from pathlib import Path

from newsmood.config import Settings
from newsmood.data.phrasebank import LABEL_NAMES, load_splits, verify_splits
from newsmood.models.classifier import load_classifier, predict


def predict_test_split(settings: Settings) -> tuple[list[str], list[str], list[str]]:
    """Returns (sentences, y_true, y_pred) over the pinned test split."""
    model_dir = Path(settings.training.output_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"No fine-tuned model in {model_dir}/. Run `newsmood train` first.")

    splits_dir = Path(settings.dataset.splits_dir)
    splits = load_splits(splits_dir)
    verify_splits(splits_dir)
    test = splits["test"]

    model, tokenizer = load_classifier(str(model_dir))

    sentences = list(test["sentence"])
    y_true = [LABEL_NAMES[i] for i in test["label"]]
    # Same truncation cap the model was trained with.
    y_pred = predict(sentences, model, tokenizer, settings.training.batch_size, "cpu", settings.training.max_length)
    return sentences, y_true, y_pred
