import pytest

from newsmood.evaluation.metrics import (
    accuracy,
    confusion_matrix,
    evaluate,
    format_eval_result,
    macro_f1,
    per_class_f1,
)

LABELS = ["negative", "neutral", "positive"]


def test_accuracy_all_correct():
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "neutral"]
    assert accuracy(y_true, y_pred) == 1.0


def test_accuracy_partial():
    y_true = ["positive", "negative", "neutral", "neutral"]
    y_pred = ["positive", "positive", "neutral", "negative"]
    assert accuracy(y_true, y_pred) == 0.5


def test_accuracy_empty_raises():
    with pytest.raises(ValueError):
        accuracy([], [])


def test_confusion_matrix_shape_and_counts():
    y_true = ["positive", "positive", "negative", "neutral"]
    y_pred = ["positive", "neutral", "negative", "neutral"]
    matrix = confusion_matrix(y_true, y_pred, LABELS)
    # rows/cols ordered negative, neutral, positive
    assert matrix == [
        [1, 0, 0],  # true negative -> pred negative
        [0, 1, 0],  # true neutral -> pred neutral
        [0, 1, 1],  # true positive -> pred neutral once, positive once
    ]


def test_per_class_f1_perfect_prediction_is_one():
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "neutral"]
    scores = per_class_f1(y_true, y_pred, LABELS)
    assert all(score == 1.0 for score in scores.values())


def test_per_class_f1_never_predicted_is_zero():
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["negative", "negative", "neutral"]
    scores = per_class_f1(y_true, y_pred, LABELS)
    assert scores["positive"] == 0.0


def test_macro_f1_is_mean_of_per_class():
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "neutral"]
    assert macro_f1(y_true, y_pred, LABELS) == 1.0


def test_macro_f1_matches_manual_average_on_imbalanced_case():
    y_true = ["neutral", "neutral", "neutral", "positive", "negative"]
    y_pred = ["neutral", "neutral", "positive", "positive", "negative"]
    per_class = per_class_f1(y_true, y_pred, LABELS)
    expected = sum(per_class.values()) / len(LABELS)
    assert macro_f1(y_true, y_pred, LABELS) == expected


def test_evaluate_bundles_all_three():
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "neutral"]
    result = evaluate(y_true, y_pred, LABELS)
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.labels == LABELS
    assert sum(sum(row) for row in result.confusion) == 3


def test_format_eval_result_includes_model_name_and_numbers():
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "neutral"]
    result = evaluate(y_true, y_pred, LABELS)
    text = format_eval_result("vader", result)
    assert "model=vader" in text
    assert "accuracy=1.0000" in text
    assert "macro_f1=1.0000" in text
