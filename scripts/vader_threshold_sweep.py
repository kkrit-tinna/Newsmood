"""Sweep VADER's positive/negative cutoffs over the pinned test split.

Rerun: `python scripts/vader_threshold_sweep.py` with newsmood_env active.

Scores the raw test sentences once, then re-thresholds the cached compounds
with the same `label_for_compound` and metric functions `newsmood eval --all`
uses. Cutoffs run -0.95..+0.95 in 0.05 steps, built from integers so there is
no float drift. Pairs with positive < negative are skipped as incoherent.
Tuning on the test split flatters VADER; this is an upper bound, not a model.
"""

from __future__ import annotations

from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from newsmood.baselines.vader import label_for_compound
from newsmood.config import get_settings
from newsmood.data.phrasebank import LABEL_NAMES, load_splits
from newsmood.evaluation.metrics import accuracy, macro_f1, per_class_recall

GRID = [round(k * 0.05, 2) for k in range(-19, 20)]

# Rows as currently printed in docs/baselines.md §Analysis, for comparison.
DOC_ROWS = [
    ("Configured", 0.05, -0.05, 0.5405, 0.4583, 0.2222),
    ("Best macro-F1 on grid", 0.40, -0.25, 0.6100, 0.4807, 0.2063),
    ("Best macro-F1 above floor", 0.50, -0.25, 0.6274, 0.4611, 0.2063),
    ("Best accuracy on grid", 0.70, -0.30, 0.6448, 0.4273, 0.1905),
    ("Symmetric ±0.60", 0.60, -0.60, 0.6178, 0.3371, 0.0000),
]


def score(compounds: list[float], y_true: list[str], pos: float, neg: float) -> dict[str, float]:
    y_pred = [label_for_compound(c, pos, neg) for c in compounds]
    return {
        "pos": pos,
        "neg": neg,
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, LABEL_NAMES),
        "neg_recall": per_class_recall(y_true, y_pred, LABEL_NAMES)["negative"],
    }


def fmt(r: dict[str, float]) -> str:
    return (
        f"{r['pos']:+.2f} / {r['neg']:+.2f}  acc {r['accuracy']:.2%}  "
        f"macro-F1 {r['macro_f1']:.2%}  neg recall {r['neg_recall']:.2%}"
    )


def main() -> None:
    settings = get_settings()
    test = load_splits(Path(settings.dataset.splits_dir))["test"]
    sentences = list(test["sentence"])
    y_true = [LABEL_NAMES[i] for i in test["label"]]

    analyzer = SentimentIntensityAnalyzer()
    compounds = [analyzer.polarity_scores(s)["compound"] for s in sentences]

    floor = max(y_true.count(label) for label in LABEL_NAMES) / len(y_true)
    configured = score(compounds, y_true, settings.vader.positive_threshold, settings.vader.negative_threshold)

    results = [score(compounds, y_true, pos, neg) for pos in GRID for neg in GRID if pos >= neg]
    print(f"n={len(y_true)}  grid values={len(GRID)}  valid pairs={len(results)}")
    print(f"majority-class floor: {floor:.2%}")
    print(f"configured: {fmt(configured)}")

    print("\n1. docs/baselines.md rows")
    for name, pos, neg, acc, mf1, nr in DOC_ROWS:
        r = score(compounds, y_true, pos, neg)
        got = (round(r["accuracy"], 4), round(r["macro_f1"], 4), round(r["neg_recall"], 4))
        status = "OK" if got == (acc, mf1, nr) else f"MISMATCH (doc: {acc:.2%} / {mf1:.2%} / {nr:.2%})"
        print(f"  {name:<22} {fmt(r)}  {status}")

    best_mf1 = max(results, key=lambda r: (r["macro_f1"], r["accuracy"]))
    best_acc = max(results, key=lambda r: (r["accuracy"], r["macro_f1"]))
    print(f"  grid argmax macro-F1: {fmt(best_mf1)}")
    print(f"  grid argmax accuracy: {fmt(best_acc)}")

    clearing = [r for r in results if r["accuracy"] > floor]
    print(f"\n2. pairs with accuracy > {floor:.2%}: {len(clearing)} of {len(results)}")

    if clearing:
        top_mf1 = max(clearing, key=lambda r: (r["macro_f1"], r["accuracy"]))
        top_nr = max(clearing, key=lambda r: (r["neg_recall"], r["macro_f1"]))
        print("\n3. among floor-clearing pairs")
        print(f"  max macro-F1:        {fmt(top_mf1)}")
        print(f"  max negative recall: {fmt(top_nr)}")

        beating = [r for r in clearing if r["macro_f1"] > configured["macro_f1"]]
        print(f"\n4. floor-clearing pairs with macro-F1 > configured {configured['macro_f1']:.2%}: {len(beating)}")
        for r in sorted(beating, key=lambda r: -r["macro_f1"]):
            print(f"  {fmt(r)}")


if __name__ == "__main__":
    main()
