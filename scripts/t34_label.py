"""T3.4: blind terminal labeler over data/sample/t34_blind.jsonl.

Shows one title at a time with its position (k/100). Starts at the first
unlabeled row. Saves atomically (temp file + rename) after every answer, so
quitting or a crash loses nothing. Reads only the blind file; the model's
labels are never opened here.

Input:
  n / u / p   negative / neutral / positive
  nh, uh, ph  the same, flagged hard
  b           back one row (re-answer it)
  q           quit
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

DEFAULT_PATH = Path("data/sample/t34_blind.jsonl")
LABELS = {"n": "negative", "u": "neutral", "p": "positive"}
SHORT = {v: k for k, v in LABELS.items()}


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save(path: Path, rows: list[dict]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse(answer: str) -> tuple[str, bool] | None:
    """'u' -> ('neutral', False), 'uh' -> ('neutral', True); None if invalid."""
    if len(answer) == 1 and answer in LABELS:
        return LABELS[answer], False
    if len(answer) == 2 and answer[0] in LABELS and answer[1] == "h":
        return LABELS[answer[0]], True
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, default=DEFAULT_PATH, help=f"blind JSONL to label (default {DEFAULT_PATH})")
    a = p.parse_args()

    rows = load(a.file)
    total = len(rows)
    i = next((k for k, r in enumerate(rows) if not r["label"]), total)
    if i == total:
        print(f"All {total} rows are labeled.")
        return

    while i < total:
        row = rows[i]
        current = ""
        if row["label"]:
            current = f"   [current: {SHORT[row['label']]}{'h' if row['hard'] else ''}]"
        print(f"\n{i + 1}/{total}{current}\n  {row['title']}")
        try:
            answer = input("n/u/p [+h], b, q > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if answer == "q":
            break
        if answer == "b":
            if i == 0:
                print("Already at the first row.")
            else:
                i -= 1
            continue
        parsed = parse(answer)
        if parsed is None:
            print("Enter n, u or p (append h for hard), b or q.")
            continue
        row["label"], row["hard"] = parsed
        save(a.file, rows)
        i += 1

    done = sum(1 for r in rows if r["label"])
    print(f"{done}/{total} labeled. Saved to {a.file}.", file=sys.stderr)


if __name__ == "__main__":
    main()
