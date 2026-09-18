"""Classification metrics: accuracy, macro-F1, confusion matrix.

Minimal set needed by T1.3's baseline eval. T1.5 extends this with
weighted-F1, per-class recall, and the majority-class / stratified-random
floor rows — see IMPLEMENTATION_GUIDE.md T1.5.
"""

from __future__ import annotations

from dataclasses import dataclass


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        raise ValueError("y_true is empty")
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    """Rows are true labels, columns are predicted labels, in `labels` order."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for true, pred in zip(y_true, y_pred):
        matrix[index[true]][index[pred]] += 1
    return matrix


def per_class_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, float]:
    scores = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores[label] = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return scores


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    scores = per_class_f1(y_true, y_pred, labels)
    return sum(scores.values()) / len(labels)


@dataclass
class EvalResult:
    accuracy: float
    macro_f1: float
    confusion: list[list[int]]
    labels: list[str]


def evaluate(y_true: list[str], y_pred: list[str], labels: list[str]) -> EvalResult:
    return EvalResult(
        accuracy=accuracy(y_true, y_pred),
        macro_f1=macro_f1(y_true, y_pred, labels),
        confusion=confusion_matrix(y_true, y_pred, labels),
        labels=labels,
    )


def format_eval_result(model_name: str, result: EvalResult) -> str:
    lines = [
        f"model={model_name}",
        f"accuracy={result.accuracy:.4f}",
        f"macro_f1={result.macro_f1:.4f}",
        "confusion matrix (rows=true, cols=pred):",
        "         " + "  ".join(f"{label:>9}" for label in result.labels),
    ]
    for label, row in zip(result.labels, result.confusion):
        lines.append(f"{label:>9}" + "".join(f"{count:>11}" for count in row))
    return "\n".join(lines)
