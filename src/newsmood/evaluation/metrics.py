"""Classification metrics: accuracy, macro-F1, weighted-F1, per-class recall,
confusion matrix.

This is what T1.3's baseline eval needs. T1.5 extends it with the
majority-class / stratified-random floor rows and assembles the
one-row-per-method table (`newsmood eval --all`) out of `EvalResult` —
see IMPLEMENTATION_GUIDE.md T1.5. `per_class_stats` is the shared core so
that extension adds rows/callers rather than rewriting the arithmetic.
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


def per_class_stats(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, dict[str, float]]:
    """Precision, recall, F1, and support (count of true instances) per class."""
    stats = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        stats[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return stats


def per_class_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, float]:
    return {label: s["f1"] for label, s in per_class_stats(y_true, y_pred, labels).items()}


def per_class_recall(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, float]:
    return {label: s["recall"] for label, s in per_class_stats(y_true, y_pred, labels).items()}


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    scores = per_class_f1(y_true, y_pred, labels)
    return sum(scores.values()) / len(labels)


def weighted_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    """F1 per class, weighted by each class's support (true-instance count)."""
    stats = per_class_stats(y_true, y_pred, labels)
    total_support = sum(s["support"] for s in stats.values())
    if total_support == 0:
        raise ValueError("y_true is empty")
    return sum(s["f1"] * s["support"] for s in stats.values()) / total_support


@dataclass
class EvalResult:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_recall: dict[str, float]
    confusion: list[list[int]]
    labels: list[str]


def evaluate(y_true: list[str], y_pred: list[str], labels: list[str]) -> EvalResult:
    return EvalResult(
        accuracy=accuracy(y_true, y_pred),
        macro_f1=macro_f1(y_true, y_pred, labels),
        weighted_f1=weighted_f1(y_true, y_pred, labels),
        per_class_recall=per_class_recall(y_true, y_pred, labels),
        confusion=confusion_matrix(y_true, y_pred, labels),
        labels=labels,
    )


def format_eval_result(model_name: str, result: EvalResult) -> str:
    lines = [
        f"model={model_name}",
        f"accuracy={result.accuracy:.4f}",
        f"macro_f1={result.macro_f1:.4f}",
        f"weighted_f1={result.weighted_f1:.4f}",
        "per_class_recall:",
    ]
    for label in result.labels:
        lines.append(f"  {label}={result.per_class_recall[label]:.4f}")
    lines += [
        "confusion matrix (rows=true, cols=pred):",
        "         " + "  ".join(f"{label:>9}" for label in result.labels),
    ]
    for label, row in zip(result.labels, result.confusion):
        lines.append(f"{label:>9}" + "".join(f"{count:>11}" for count in row))
    return "\n".join(lines)
