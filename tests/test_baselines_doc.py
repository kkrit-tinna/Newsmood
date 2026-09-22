from pathlib import Path

from newsmood.evaluation.baselines_doc import (
    _END_MARKER,
    _START_MARKER,
    render_summary_table,
    update_baselines_doc,
)

_TABLE = {
    "majority_class": {
        "accuracy": 0.621,
        "macro_f1": 0.255,
        "weighted_f1": 0.475,
        "per_class_recall": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
        "confusion": [[0, 63, 0], [0, 322, 0], [0, 133, 0]],
        "labels": ["negative", "neutral", "positive"],
    },
    "vader": {
        "accuracy": 0.5405,
        "macro_f1": 0.4583,
        "weighted_f1": 0.5529,
        "per_class_recall": {"negative": 0.2222, "neutral": 0.5497, "positive": 0.6692},
        "confusion": [[14, 17, 32], [19, 177, 126], [11, 33, 89]],
        "labels": ["negative", "neutral", "positive"],
    },
}

_HUMAN_CEILING_NOTE = "25% of annotators disagreed; treat high scores as brushing a floor, not a target."
_SEED = 42


def test_render_summary_table_has_one_row_per_method_plus_human_ceiling():
    text = render_summary_table(_TABLE, _HUMAN_CEILING_NOTE, _SEED)
    assert "Majority class" in text
    assert "VADER" in text
    assert "Human ceiling" in text
    assert _HUMAN_CEILING_NOTE in text


def test_render_summary_table_reports_accuracy_macro_f1_weighted_f1_and_recall():
    text = render_summary_table(_TABLE, _HUMAN_CEILING_NOTE, _SEED)
    assert "54.05%" in text  # vader accuracy
    assert "45.83%" in text  # vader macro-F1
    assert "55.29%" in text  # vader weighted-F1
    assert "22.22%" in text  # vader negative recall


def test_render_summary_table_is_wrapped_in_markers():
    text = render_summary_table(_TABLE, _HUMAN_CEILING_NOTE, _SEED)
    assert text.startswith(_START_MARKER)
    assert text.endswith(_END_MARKER)


def test_render_summary_table_names_the_test_split_population():
    text = render_summary_table(_TABLE, _HUMAN_CEILING_NOTE, _SEED)
    # confusion matrices in _TABLE sum to 518 (63 + 322 + 133)
    assert "n=518" in text


def test_render_summary_table_discloses_stratified_random_is_a_single_seeded_draw():
    text = render_summary_table(_TABLE, _HUMAN_CEILING_NOTE, seed=42)
    assert "single seeded draw" in text
    assert "dataset.seed=42" in text


def test_update_baselines_doc_creates_file_when_missing(tmp_path):
    path = tmp_path / "baselines.md"
    update_baselines_doc(path, _TABLE, _HUMAN_CEILING_NOTE, _SEED)
    content = path.read_text()
    assert content.startswith("# Baselines")
    assert "VADER" in content
    assert _START_MARKER in content and _END_MARKER in content


def test_update_baselines_doc_preserves_hand_written_prose_below_the_table(tmp_path):
    path = tmp_path / "baselines.md"
    path.write_text(
        "# Baselines\n\n## VADER\n\nSome hand-written failure-mode analysis that must survive.\n"
    )

    update_baselines_doc(path, _TABLE, _HUMAN_CEILING_NOTE, _SEED)

    content = path.read_text()
    assert "Some hand-written failure-mode analysis that must survive." in content
    assert "## VADER" in content


def test_update_baselines_doc_is_idempotent_on_rerun(tmp_path):
    path = tmp_path / "baselines.md"
    path.write_text("# Baselines\n\n## VADER\n\nHand-written analysis.\n")

    update_baselines_doc(path, _TABLE, _HUMAN_CEILING_NOTE, _SEED)
    first = path.read_text()
    update_baselines_doc(path, _TABLE, _HUMAN_CEILING_NOTE, _SEED)
    second = path.read_text()

    assert first == second
    assert "Hand-written analysis." in second
