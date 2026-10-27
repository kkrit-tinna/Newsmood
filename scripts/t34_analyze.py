"""T3.4: score the model against the frozen hand labels.

Reads (never writes) three files:
  data/sample/hand_labeled_100.jsonl   id, title, label, hard (frozen)
  data/t34_model_labels.jsonl          id, source, published_at, label, confidence
  data/t34_relabel_20.jsonl            blind relabel of 20 ids

Prints the tables to stdout and writes the same results to
data/t34_analysis.json (gitignored), so drift.md can be checked against it.
Exits 1 if the files disagree on ids or carry an unknown label.
"""

import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

HAND_PATH = Path("data/sample/hand_labeled_100.jsonl")
MODEL_PATH = Path("data/t34_model_labels.jsonl")
RELABEL_PATH = Path("data/t34_relabel_20.jsonl")
OUT_PATH = Path("data/t34_analysis.json")
LABELS = ("negative", "neutral", "positive")
Z95 = 1.959963984540054
# DistilBERT, T2.3, sentences_75agree test split (docs/baselines.md).
REFERENCE = {"model": "DistilBERT (fine-tuned)", "task": "T2.3",
             "split": "PhraseBank sentences_75agree test", "n": 518,
             "accuracy": 0.9208}


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                sys.exit(f"error: duplicate id {row['id'][:12]} in {path}")
            if row["label"] not in LABELS:
                sys.exit(f"error: unknown label {row['label']!r} in {path}")
            rows[row["id"]] = row
    return rows


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    denom = 1 + Z95**2 / n
    centre = (p + Z95**2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def accuracy(pairs: list[tuple[str, str]]) -> dict:
    n = len(pairs)
    k = sum(a == b for a, b in pairs)
    lo, hi = wilson(k, n)
    return {"n": n, "correct": k, "accuracy": k / n, "wilson95": [lo, hi]}


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    p_o = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    p_e = sum(first[c] * second[c] for c in LABELS) / n**2
    return None if p_e == 1 else (p_o - p_e) / (1 - p_e)


def write_json(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def fmt_acc(name: str, a: dict) -> str:
    lo, hi = a["wilson95"]
    return (f"  {name:<28} {a['correct']:>3}/{a['n']:<3} {pct(a['accuracy']):>7}"
            f"   95% CI [{pct(lo)}, {pct(hi)}]")


def main() -> None:
    hand = read_jsonl(HAND_PATH)
    model = read_jsonl(MODEL_PATH)
    relabel = read_jsonl(RELABEL_PATH)
    if set(hand) != set(model):
        sys.exit("error: hand and model files cover different ids")
    if not set(relabel) <= set(hand):
        sys.exit("error: relabel has ids not in the hand-labeled 100")

    ids = sorted(hand)

    # Self-agreement on the blind relabel.
    rel_ids = sorted(relabel)
    rel_pairs = [(hand[i]["label"], relabel[i]["label"]) for i in rel_ids]
    changed = [{"id": i, "title": hand[i]["title"], "original": hand[i]["label"],
                "relabel": relabel[i]["label"], "hard_original": hand[i]["hard"]}
               for i in rel_ids if hand[i]["label"] != relabel[i]["label"]]
    self_agreement = {"n": len(rel_pairs),
                      "agree": sum(a == b for a, b in rel_pairs),
                      "raw": sum(a == b for a, b in rel_pairs) / len(rel_pairs),
                      "cohen_kappa": cohen_kappa(rel_pairs),
                      "changed": changed}

    # Model vs hand labels.
    pairs = {i: (hand[i]["label"], model[i]["label"]) for i in ids}
    acc_all = accuracy(list(pairs.values()))
    acc_easy = accuracy([pairs[i] for i in ids if not hand[i]["hard"]])
    acc_hard = accuracy([pairs[i] for i in ids if hand[i]["hard"]])

    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    for t, p in pairs.values():
        confusion[t][p] += 1
    recall = {}
    for t in LABELS:
        n = sum(confusion[t].values())
        recall[t] = {"n": n, "correct": confusion[t][t],
                     "recall": confusion[t][t] / n if n else None}

    by_source = {}
    for src in sorted({model[i]["source"] for i in ids}):
        by_source[src] = accuracy([pairs[i] for i in ids
                                   if model[i]["source"] == src])

    disagreements = sorted(
        ({"id": i, "source": model[i]["source"], "title": hand[i]["title"],
          "hand": hand[i]["label"], "model": model[i]["label"],
          "confidence": model[i]["confidence"], "hard": hand[i]["hard"]}
         for i in ids if pairs[i][0] != pairs[i][1]),
        key=lambda d: (d["hand"], d["model"], -d["confidence"]))

    results = {"inputs": {"hand": str(HAND_PATH), "model": str(MODEL_PATH),
                          "relabel": str(RELABEL_PATH)},
               "reference": REFERENCE,
               "self_agreement": self_agreement,
               "accuracy": {"all": acc_all, "not_hard": acc_easy,
                            "hard": acc_hard},
               "confusion": {"rows": "hand", "cols": "model",
                             "matrix": confusion},
               "recall": recall,
               "by_source": by_source,
               "disagreements": disagreements}
    write_json(OUT_PATH, results)

    sa = self_agreement
    kappa = "undefined" if sa["cohen_kappa"] is None else f"{sa['cohen_kappa']:.3f}"
    print(f"Self-agreement (blind relabel): {sa['agree']}/{sa['n']} "
          f"= {pct(sa['raw'])}, Cohen's kappa = {kappa}")
    for c in changed:
        print(f"  {c['id'][:12]}  {c['original']:>8} -> {c['relabel']:<8} "
              f"hard={c['hard_original']}  {c['title']}")

    print("\nAccuracy vs hand labels")
    print(fmt_acc("All", acc_all))
    print(fmt_acc("hard == false", acc_easy))
    print(fmt_acc("hard == true", acc_hard))
    print(f"  {'Reference: T2.3 PhraseBank':<28} {'':>7} {pct(REFERENCE['accuracy']):>7}"
          f"   (test split, n={REFERENCE['n']})")

    print("\nConfusion (rows = hand, cols = model)")
    print(f"  {'':>10}" + "".join(f"{p:>10}" for p in LABELS) + f"{'n':>6}")
    for t in LABELS:
        print(f"  {t:>10}" + "".join(f"{confusion[t][p]:>10}" for p in LABELS)
              + f"{sum(confusion[t].values()):>6}")

    print("\nPer-class recall")
    for t in LABELS:
        r = recall[t]
        val = "n/a" if r["recall"] is None else pct(r["recall"])
        print(f"  {t:<10} {r['correct']:>3}/{r['n']:<3} {val:>7}")

    print("\nAccuracy by source")
    for src, a in by_source.items():
        print(fmt_acc(src, a))

    print(f"\nDisagreements ({len(disagreements)})")
    for d in disagreements:
        print(f"  {d['id'][:12]}  {d['source']:<18} hand={d['hand']:<8} "
              f"model={d['model']:<8} conf={d['confidence']:.3f} "
              f"hard={str(d['hard']):<5}  {d['title']}")

    print(f"\nwrote {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
