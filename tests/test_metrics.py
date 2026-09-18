import pytest

from newsmood.evaluation.metrics import (
    accuracy,
    confusion_matrix,
    evaluate,
    format_eval_result,
    macro_f1,
    per_class_f1,
    per_class_recall,
    per_class_stats,
    weighted_f1,
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


# --- per_class_stats: the shared core behind f1/recall/weighted_f1 ---


def test_per_class_stats_support_counts_true_instances():
    y_true = ["neutral", "neutral", "neutral", "positive", "negative"]
    y_pred = ["neutral", "neutral", "positive", "positive", "negative"]
    stats = per_class_stats(y_true, y_pred, LABELS)
    assert stats["neutral"]["support"] == 3
    assert stats["positive"]["support"] == 1
    assert stats["negative"]["support"] == 1


def test_per_class_stats_precision_recall_f1_on_known_case():
    # neutral: predicted twice, both correct (tp=2, fp=0); 3 true instances,
    # one (index 2) predicted as positive instead (fn=1)
    y_true = ["neutral", "neutral", "neutral", "positive", "negative"]
    y_pred = ["neutral", "neutral", "positive", "positive", "negative"]
    stats = per_class_stats(y_true, y_pred, LABELS)
    assert stats["neutral"]["precision"] == pytest.approx(1.0)
    assert stats["neutral"]["recall"] == pytest.approx(2 / 3)


# --- per_class_recall ---


def test_per_class_recall_perfect_prediction_is_one():
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "neutral"]
    scores = per_class_recall(y_true, y_pred, LABELS)
    assert all(score == 1.0 for score in scores.values())


def test_per_class_recall_never_recovered_is_zero():
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["negative", "negative", "neutral"]
    scores = per_class_recall(y_true, y_pred, LABELS)
    assert scores["positive"] == 0.0


# --- weighted_f1 ---


def test_weighted_f1_perfect_prediction_is_one():
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "neutral"]
    assert weighted_f1(y_true, y_pred, LABELS) == 1.0


def test_weighted_f1_matches_manual_support_weighted_average():
    y_true = ["neutral", "neutral", "neutral", "positive", "negative"]
    y_pred = ["neutral", "neutral", "positive", "positive", "negative"]
    stats = per_class_stats(y_true, y_pred, LABELS)
    total_support = sum(s["support"] for s in stats.values())
    expected = sum(s["f1"] * s["support"] for s in stats.values()) / total_support
    assert weighted_f1(y_true, y_pred, LABELS) == pytest.approx(expected)


def test_weighted_f1_empty_raises():
    with pytest.raises(ValueError):
        weighted_f1([], [], LABELS)


def test_weighted_f1_differs_from_macro_f1_when_classes_imbalanced():
    # majority class predicted perfectly, minority classes missed entirely --
    # weighted should track the majority class more closely than macro does
    y_true = ["neutral"] * 8 + ["positive", "negative"]
    y_pred = ["neutral"] * 8 + ["neutral", "neutral"]
    assert weighted_f1(y_true, y_pred, LABELS) > macro_f1(y_true, y_pred, LABELS)


# --- evaluate / format_eval_result ---


def test_evaluate_bundles_all_metrics():
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "neutral"]
    result = evaluate(y_true, y_pred, LABELS)
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.weighted_f1 == 1.0
    assert result.per_class_recall == {label: 1.0 for label in LABELS}
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
    assert "weighted_f1=1.0000" in text
    assert "negative=1.0000" in text
    assert "neutral=1.0000" in text
    assert "positive=1.0000" in text
