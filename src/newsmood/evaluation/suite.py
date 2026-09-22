"""Assembles the one-row-per-method table `newsmood eval --all` writes to
docs/baselines.md (IMPLEMENTATION_GUIDE.md T1.5). T2.3 adds DistilBERT and
FinBERT rows to the same table the same way — one more
`calculate_comprehensive_metrics` call merged in — and T3.5's quality gates
read the merged table rather than reimplementing any of this.
"""

from __future__ import annotations

from newsmood.baselines import logreg, reference, vader
from newsmood.config import Settings
from newsmood.data.phrasebank import LABEL_NAMES
from newsmood.evaluation.metrics import calculate_comprehensive_metrics

HUMAN_CEILING_NOTE = (
    "Not a number to compare against directly: `sentences_75agree` keeps only "
    "rows where at least 75% of annotators agreed, so up to 25% of annotators "
    "disagreed with the kept label on every one of them. A model scoring in "
    "the high-80s/low-90s may be brushing a ceiling inherent to the labels, "
    "not still leaving headroom on the table."
)

# method key -> (predict_test_split callable, requires the `[train]` extra)
_RUNS = {
    "majority_class": reference.majority_class_predict_test_split,
    "stratified_random": reference.stratified_random_predict_test_split,
    "vader": vader.predict_test_split,
    "logreg": logreg.predict_test_split,
}


def run_reference_and_baselines(settings: Settings) -> dict[str, dict]:
    """Runs majority-class, stratified-random, VADER, and LogReg against the
    pinned test split and returns one merged metrics table, method name ->
    `calculate_comprehensive_metrics` row."""
    table: dict[str, dict] = {}
    for method, predict_test_split in _RUNS.items():
        _, y_true, y_pred = predict_test_split(settings)
        table.update(calculate_comprehensive_metrics(method, y_true, y_pred, LABEL_NAMES))
    return table
