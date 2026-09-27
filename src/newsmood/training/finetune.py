"""DistilBERT fine-tuning on the pinned PhraseBank split.

`get_device()` and `EarlyStopping` are ported from the archived repo
(IMPLEMENTATION_GUIDE.md §3). EarlyStopping's checkpoint copy is fixed, not
ported as-is — §9 Deviations, 2026-09-27.

The model is `AutoModelForSequenceClassification`, not a custom nn.Module, so
the saved artifact loads with `from_pretrained` without importing this package.

`fine_tune()` loads the pinned split and the base model; `train_loop()` is the
loop itself, taking a model and tokenizer so tests can drive it with a tiny one.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from newsmood.config import Settings, TrainingSettings
from newsmood.data.phrasebank import LABEL_NAMES, load_splits, verify_splits
from newsmood.evaluation.metrics import EvalResult, accuracy, evaluate
from newsmood.training.dataset import SentimentDataset

ID2LABEL = dict(enumerate(LABEL_NAMES))
LABEL2ID = {name: i for i, name in ID2LABEL.items()}


# Order changed from the archived mps > cuda > cpu. §9 Deviations, 2026-09-27.
def get_device():
    """Detect and return the best available device for training"""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("   Using NVIDIA GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("   Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("   Using CPU")

    return device


DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")


def resolve_device(choice: str) -> torch.device:
    """`auto` defers to get_device(). An explicit choice overrides it, and
    fails here rather than mid-run if that backend is not available."""
    if choice == "auto":
        return get_device()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("--device mps requested but MPS is not available on this machine")
        return torch.device("mps")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is not available on this machine")
        return torch.device("cuda")
    raise ValueError(f"unknown device {choice!r}, expected one of {DEVICE_CHOICES}")


def seed_everything(seed: int) -> None:
    """Seed Python, numpy and torch. torch.manual_seed covers CPU, CUDA and
    MPS generators. Call before the model is built: the classifier head's
    initial weights come from torch's RNG."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        # The archived repo used state_dict().copy(), a shallow copy whose
        # tensors alias the live parameters: training kept overwriting the
        # "best" weights and the restore was a silent no-op.
        self.best_weights = copy.deepcopy(model.state_dict())


def class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Balanced weights n / (k * count_c), the same formula as sklearn's
    class_weight="balanced". Pass the full training split's labels."""
    counts = Counter(labels)
    missing = [c for c in range(num_classes) if counts[c] == 0]
    if missing:
        raise ValueError(f"No training rows for class ids {missing}; cannot weight the loss.")
    n = len(labels)
    return torch.tensor([n / (num_classes * counts[c]) for c in range(num_classes)], dtype=torch.float32)


def build_model(settings: Settings):
    # dataset.tokenizer names the base model too: max_length was measured
    # under that tokenizer, so a second key could drift from it.
    return AutoModelForSequenceClassification.from_pretrained(
        settings.dataset.tokenizer,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


@dataclass
class DryRunResult:
    n_rows: int
    padded_length: int
    device: str
    seed: int
    loss: float
    weights: list[float]
    n_params_with_grad: int

    def summary(self) -> str:
        weights = ", ".join(f"{name}={w:.3f}" for name, w in zip(LABEL_NAMES, self.weights))
        return (
            f"dry run ok on {self.device}, seed {self.seed}: {self.n_rows} rows, padded to {self.padded_length} tokens, "
            f"loss {self.loss:.4f}, gradients on {self.n_params_with_grad} parameter tensors\n"
            f"class weights (train split): {weights}"
        )


def dry_run(settings: Settings, limit: int, device: torch.device) -> DryRunResult:
    """One forward and one backward pass over the first `limit` training rows.

    Weights come from the whole training split, not the `limit` subset. Saves
    nothing. Raises if any trainable parameter received no gradient.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    seed_everything(settings.dataset.seed)

    splits_dir = Path(settings.dataset.splits_dir)
    splits = load_splits(splits_dir)
    verify_splits(splits_dir)
    train = splits["train"]

    weights = class_weights(list(train["label"]), len(LABEL_NAMES))
    subset = train.select(range(min(limit, len(train))))

    tokenizer = AutoTokenizer.from_pretrained(settings.dataset.tokenizer)
    dataset = SentimentDataset(list(subset["sentence"]), list(subset["label"]), tokenizer, settings.training.max_length)
    loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=DataCollatorWithPadding(tokenizer))
    batch = {k: v.to(device) for k, v in next(iter(loader)).items()}

    model = build_model(settings).to(device)
    model.train()
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    loss = nn.CrossEntropyLoss(weight=weights.to(device))(logits, batch["labels"])
    loss.backward()

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    no_grad = [name for name, p in trainable if p.grad is None]
    if no_grad:
        raise RuntimeError(f"Backward pass left {len(no_grad)} parameters without a gradient, e.g. {no_grad[:3]}")

    return DryRunResult(
        n_rows=len(dataset),
        padded_length=batch["input_ids"].shape[1],
        device=str(device),
        seed=settings.dataset.seed,
        loss=loss.item(),
        weights=weights.tolist(),
        n_params_with_grad=len(trainable),
    )


def _names(ids: list[int]) -> list[str]:
    return [LABEL_NAMES[i] for i in ids]


def _loader(texts, labels, tokenizer, max_length, batch_size, shuffle_seed=None) -> DataLoader:
    """shuffle_seed=None means no shuffling (evaluation order)."""
    dataset = SentimentDataset(texts, labels, tokenizer, max_length)
    generator = None if shuffle_seed is None else torch.Generator().manual_seed(shuffle_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_seed is not None,
        generator=generator,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


def _forward(model, batch, criterion, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    return logits, criterion(logits, batch["labels"]), batch["labels"]


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, desc: str) -> float:
    """Returns the row-weighted mean training loss. The progress bar is closed
    when this returns, so the caller's epoch summary prints below it."""
    model.train()
    total, n = 0.0, 0
    with tqdm(loader, desc=desc, unit="batch") as bar:
        for batch in bar:
            _, loss, labels = _forward(model, batch, criterion, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(labels)
            n += len(labels)
            bar.set_postfix(loss=f"{total / n:.4f}")
    return total / n


@torch.no_grad()
def evaluate_split(model, loader, criterion, device) -> tuple[float, EvalResult]:
    """Row-weighted mean loss (same class-weighted criterion as training) and
    evaluation.metrics' EvalResult over the loader."""
    model.eval()
    total, n = 0.0, 0
    y_true, y_pred = [], []
    for batch in loader:
        logits, loss, labels = _forward(model, batch, criterion, device)
        total += loss.item() * len(labels)
        n += len(labels)
        y_true += labels.tolist()
        y_pred += logits.argmax(dim=-1).tolist()
    return total / n, evaluate(_names(y_true), _names(y_pred), LABEL_NAMES)


def write_history(path: Path, history: dict) -> None:
    """Write via a temp file and rename, so a crash mid-write leaves the
    previous epoch's history intact rather than a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(history, indent=2) + "\n")
    os.replace(tmp, path)


def format_epoch(record: dict, n_epochs: int) -> str:
    return (
        f"epoch {record['epoch']}/{n_epochs}  "
        f"train_loss {record['train_loss']:.4f}  val_loss {record['val_loss']:.4f}  "
        f"val_acc {record['val_accuracy']:.2%}  val_macro_f1 {record['val_macro_f1']:.2%}  "
        f"{record['seconds']:.1f}s"
    )


def train_loop(
    model,
    tokenizer,
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    hp: TrainingSettings,
    device: torch.device,
    seed: int,
    run_info: dict,
) -> dict:
    """Fine-tune `model` in place, write history.json after every epoch, then
    save_pretrained the model and tokenizer to hp.output_dir. Returns the history.

    The saved weights are the last epoch's unless EarlyStopping fires, in which
    case it restores the best-val-loss weights first.
    """
    output_dir = Path(hp.output_dir)
    history_path = output_dir / "history.json"

    weights = class_weights(train_labels, len(LABEL_NAMES))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    train_loader = _loader(train_texts, train_labels, tokenizer, hp.max_length, hp.batch_size, shuffle_seed=seed)
    val_loader = _loader(val_texts, val_labels, tokenizer, hp.max_length, hp.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    total_steps = len(train_loader) * hp.epochs
    warmup_steps = int(hp.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    stopper = EarlyStopping(patience=hp.early_stopping_patience, min_delta=hp.early_stopping_min_delta)

    # The floor to beat on val: predict the training split's majority class.
    majority = Counter(train_labels).most_common(1)[0][0]
    val_floor = accuracy(_names(val_labels), _names([majority] * len(val_labels)))

    history = {
        **run_info,
        "device": str(device),
        "seed": seed,
        "status": "running",
        "hyperparameters": {**hp.model_dump(), "total_steps": total_steps, "warmup_steps": warmup_steps},
        "class_weights": dict(zip(LABEL_NAMES, weights.tolist())),
        "train_rows": len(train_texts),
        "val_rows": len(val_texts),
        "val_majority_accuracy": val_floor,
        "epochs": [],
    }
    write_history(history_path, history)
    print(
        f"{len(train_texts)} train / {len(val_texts)} val rows, {total_steps} steps ({warmup_steps} warmup), "
        f"val majority-class floor {val_floor:.2%}"
    )

    for epoch in range(1, hp.epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, desc=f"epoch {epoch}/{hp.epochs}"
        )
        val_loss, val = evaluate_split(model, val_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val.accuracy,
            "val_macro_f1": val.macro_f1,
            "seconds": time.perf_counter() - start,
        }
        history["epochs"].append(record)
        write_history(history_path, history)
        print(format_epoch(record, hp.epochs))

        if stopper(val_loss, model):
            history["early_stopped_after_epoch"] = epoch
            print(f"early stopping after epoch {epoch}; restored weights with val_loss {stopper.best_loss:.4f}")
            break

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    history["status"] = "complete"
    history["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_history(history_path, history)
    print(f"saved model + tokenizer to {output_dir}/, history to {history_path}")
    return history


def fine_tune(settings: Settings, device: torch.device) -> dict:
    """Fine-tune the base model on the pinned train split, validate on val.
    The test split is not touched — that is T2.3."""
    seed = settings.dataset.seed
    seed_everything(seed)

    splits_dir = Path(settings.dataset.splits_dir)
    splits = load_splits(splits_dir)
    verify_splits(splits_dir)
    train, val = splits["train"], splits["val"]

    tokenizer = AutoTokenizer.from_pretrained(settings.dataset.tokenizer)
    model = build_model(settings).to(device)

    run_info = {
        "base_model": settings.dataset.tokenizer,
        "phrasebank_config": settings.dataset.phrasebank_config,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return train_loop(
        model,
        tokenizer,
        list(train["sentence"]),
        list(train["label"]),
        list(val["sentence"]),
        list(val["label"]),
        settings.training,
        device,
        seed,
        run_info,
    )
