"""Load a sequence classifier and run batched inference over raw sentences.

Shared by every transformer row in the eval table (fine-tuned DistilBERT,
FinBERT). Outputs are mapped to label names through the checkpoint's own
id2label, never by index: checkpoints disagree on order.

torch/transformers are imported inside the functions: importing torch takes
seconds, and evaluation/suite.py imports this module's callers unconditionally.
"""

from __future__ import annotations

from newsmood.data.phrasebank import LABEL_NAMES


class IncompleteCheckpoint(RuntimeError):
    """from_pretrained initialised weights the checkpoint did not supply."""


def load_classifier(name_or_path: str, revision: str | None = None):
    """Return (model, tokenizer). Raises IncompleteCheckpoint if any weight was
    missing from the checkpoint: a freshly initialised classifier head scores
    as a random model while looking like a loaded one."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model, info = AutoModelForSequenceClassification.from_pretrained(
        name_or_path, revision=revision, output_loading_info=True
    )
    if info["missing_keys"] or info["mismatched_keys"]:
        raise IncompleteCheckpoint(
            f"{name_or_path}: missing {sorted(info['missing_keys'])}, mismatched {info['mismatched_keys']}"
        )
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, revision=revision)
    return model, tokenizer


def output_labels(id2label: dict[int, str]) -> list[str]:
    """Label name for each output index, from the checkpoint's own config.

    Raises unless the checkpoint names exactly our three labels, so a revision
    that renamed or reordered them fails here instead of scoring as garbage.
    """
    names = [id2label[i].lower() for i in range(len(id2label))]
    if sorted(names) != sorted(LABEL_NAMES):
        raise ValueError(f"checkpoint id2label {id2label} does not name exactly {LABEL_NAMES}")
    return names


def classify_batch(
    texts: list[str], model, tokenizer, device, batch_size: int, max_length: int | None = None
) -> list[tuple[str, float]]:
    """(label, softmax confidence) per text, in input order. The one forward-pass
    path: classify() and predict() both go through here.

    Each batch is padded to its own longest row (padding=True), not to
    max_length, so a batch of short headlines costs short-headline compute.
    The attention mask keeps padding from changing any row's prediction.
    max_length=None truncates only at the tokenizer's own limit.
    """
    import torch
    names = output_labels(model.config.id2label)
    model.to(device)
    model.eval()
    results: list[tuple[str, float]] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            confidence, index = model(**batch).logits.softmax(dim=-1).max(dim=-1)
            # .tolist() copies to the CPU, which also waits for an MPS/CUDA
            # batch to finish, so a caller timing this call times real work.
            results += [(names[i], c) for i, c in zip(index.tolist(), confidence.tolist())]
    return results


def classify(text: str, model, tokenizer, device, max_length: int | None = None) -> tuple[str, float]:
    """(label, softmax confidence) for one headline (`score --text`)."""
    return classify_batch([text], model, tokenizer, device, batch_size=1, max_length=max_length)[0]


def predict(sentences: list[str], model, tokenizer, batch_size: int, device, max_length: int | None = None) -> list[str]:
    """Raw sentences in, label names out, for the eval table."""
    return [label for label, _ in classify_batch(sentences, model, tokenizer, device, batch_size, max_length)]
