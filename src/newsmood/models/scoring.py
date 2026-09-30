"""`newsmood score` (T2.5): score the SQLite store, or a JSONL file of sentences.

Every mode takes a BatchClassifier — texts in, (label, confidence) out, in
input order — so tests inject a fake and never download the model. Only
score_store() writes, and only to unscored rows.
"""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from newsmood.config import Settings
from newsmood.data import store
from newsmood.models.classifier import classify_batch, load_classifier
from newsmood.models.loader import ensure_model
from newsmood.training import finetune

BatchClassifier = Callable[[list[str]], list[tuple[str, float]]]


class InputFileError(ValueError):
    """A --file line that is not a JSON object with a string "sentence"."""


def resolve_device(choice: str):
    """finetune.resolve_device(), with get_device()'s notice sent to stderr:
    stdout carries results only, and --file output is parsed line by line."""
    with redirect_stdout(sys.stderr):
        return finetune.resolve_device(choice)


def load_batch_classifier(settings: Settings, device) -> BatchClassifier:
    """The published model at its pinned revision, batched per config."""
    model, tokenizer = load_classifier(str(ensure_model(settings.model)))
    batch_size, max_length = settings.model.batch_size, settings.training.max_length
    return lambda texts: classify_batch(texts, model, tokenizer, device, batch_size, max_length)


def score_store(conn, get_classifier: Callable[[], BatchClassifier], batch_size: int) -> int:
    """Score every unscored row, one transaction per batch, and return how
    many were scored. A second run finds nothing and scores 0.

    get_classifier is called only if there is something to score, so a run
    with nothing new loads no model and never triggers a download.
    """
    rows = store.unscored(conn)
    if not rows:
        return 0
    classify = get_classifier()
    scored = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        predictions = classify([title for _, title in batch])
        scores = [(id_, label, confidence) for (id_, _), (label, confidence) in zip(batch, predictions, strict=True)]
        scored += store.write_scores(conn, scores, datetime.now(timezone.utc))
    return scored


def read_jsonl(path: Path) -> list[dict]:
    """One JSON object per line, each with a string "sentence". Blank lines
    are skipped; anything else malformed raises InputFileError naming the line."""
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise InputFileError(f"{path}:{lineno}: not valid JSON ({e.msg})") from None
            if not isinstance(record, dict) or "sentence" not in record:
                raise InputFileError(f'{path}:{lineno}: missing "sentence" field')
            if not isinstance(record["sentence"], str):
                raise InputFileError(f'{path}:{lineno}: "sentence" is not a string')
            records.append(record)
    return records


def score_records(records: list[dict], classify: BatchClassifier) -> list[dict]:
    """Each record unchanged, plus pred_label and confidence. A gold "label"
    already in the record is kept as-is."""
    predictions = classify([r["sentence"] for r in records])
    return [
        {**r, "pred_label": label, "confidence": confidence}
        for r, (label, confidence) in zip(records, predictions, strict=True)
    ]
