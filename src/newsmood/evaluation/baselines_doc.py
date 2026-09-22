"""Renders the `newsmood eval --all` summary table into `docs/baselines.md`.

`docs/baselines.md` also carries hand-written failure-mode analysis (T1.3's
VADER walkthrough, T1.6's writeup) below the summary table. Writing the
whole file from this table would erase that prose every time the command
reruns, so this only replaces the region between the marker comments,
leaving everything else in the file untouched — the same "safe to rerun"
property the rest of the pipeline holds (CLAUDE.md, "every stage is
re-runnable").
"""

from __future__ import annotations

from pathlib import Path

_START_MARKER = "<!-- eval:summary:start -->"
_END_MARKER = "<!-- eval:summary:end -->"

_DISPLAY_NAMES = {
    "majority_class": "Majority class",
    "stratified_random": "Stratified random",
    "vader": "VADER",
    "logreg": "Logistic Regression",
    "distilbert": "DistilBERT (fine-tuned)",
    "finbert": "FinBERT (zero-shot)",
}

# Reference rows carry no accuracy/F1 of their own — they're a qualitative
# note appended after the measured rows.
_QUALITATIVE_ROWS = ["human_ceiling"]


def _display_name(method: str) -> str:
    return _DISPLAY_NAMES.get(method, method.replace("_", " ").title())


def _pct(value: float) -> str:
    return f"{value:.2%}"


def render_summary_table(table: dict[str, dict], human_ceiling_note: str) -> str:
    """`table` is the merged output of `calculate_comprehensive_metrics` calls
    (`evaluation/suite.py`). Row order follows insertion order in `table`."""
    lines = [
        _START_MARKER,
        "| Method | Accuracy | Macro-F1 | Weighted-F1 | Recall (negative / neutral / positive) |",
        "|---|---|---|---|---|",
    ]
    for method, metrics in table.items():
        recall = metrics["per_class_recall"]
        recall_str = " / ".join(_pct(recall[label]) for label in metrics["labels"])
        lines.append(
            f"| {_display_name(method)} | {_pct(metrics['accuracy'])} | "
            f"{_pct(metrics['macro_f1'])} | {_pct(metrics['weighted_f1'])} | {recall_str} |"
        )
    lines.append(f"| Human ceiling | — | — | — | {human_ceiling_note} |")
    lines.append(_END_MARKER)
    return "\n".join(lines)


def update_baselines_doc(path: Path, table: dict[str, dict], human_ceiling_note: str) -> None:
    """Insert or replace the summary table in `path`, leaving the rest of the
    file (hand-written analysis) alone. Creates the file with a bare title
    if it doesn't exist yet."""
    new_table = render_summary_table(table, human_ceiling_note)

    if path.exists():
        content = path.read_text()
    else:
        content = "# Baselines\n"

    if _START_MARKER in content and _END_MARKER in content:
        start = content.index(_START_MARKER)
        end = content.index(_END_MARKER) + len(_END_MARKER)
        content = content[:start] + new_table + content[end:]
    else:
        title, _, rest = content.partition("\n")
        rest = rest.lstrip("\n")
        content = f"{title}\n\n{new_table}\n" + (f"\n{rest}" if rest else "")

    path.write_text(content)
