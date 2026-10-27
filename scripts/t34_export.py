"""T3.4: export the blind 100-headline sample and the frozen model labels.

Pool: scored rows with published_at >= the store's first fetch (excludes
Yahoo's stale pre-history items). Sorted by id, then a seeded sample and a
shuffle with the same RNG, so the same store gives the same sample.

Writes two files:
  data/sample/t34_blind.jsonl     id, title, label "", hard false (committed)
  data/t34_model_labels.jsonl     id, source, published_at, label, confidence
                                  (gitignored; not opened until labels freeze)
Prints counts and paths only; never a title or a label.
"""

import argparse
import json
import os
import random
import sys
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from newsmood.config import get_settings
from newsmood.data import store

SEED = 42
SAMPLE_SIZE = 100
BLIND_PATH = Path("data/sample/t34_blind.jsonl")
MODEL_PATH = Path("data/t34_model_labels.jsonl")
END = datetime(9999, 1, 1, tzinfo=timezone.utc)  # open upper bound


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def has_labels(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open() as f:
        return any(json.loads(line).get("label") for line in f if line.strip())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="Overwrite a blind file that already holds hand labels.")
    a = p.parse_args()

    if has_labels(BLIND_PATH) and not a.force:
        sys.exit(f"error: {BLIND_PATH} already holds hand labels; refusing to overwrite (use --force)")

    db_path = get_settings().store.db_path
    with closing(store.connect(db_path)) as conn:
        start = store.first_fetched_at(conn)
        if start is None:
            sys.exit(f"error: {db_path} is empty")
        missing = store.unscored_between(conn, start, END)
        if missing:
            sys.exit(f"error: {missing} rows in the pool window are unscored; run `newsmood score` first")
        pool = store.scored_headlines_between(conn, start, END)

    if len(pool) < SAMPLE_SIZE:
        sys.exit(f"error: pool has {len(pool)} rows, need {SAMPLE_SIZE}")

    pool.sort(key=lambda r: r["id"])
    rng = random.Random(SEED)
    sample = rng.sample(pool, SAMPLE_SIZE)
    rng.shuffle(sample)

    write_jsonl(BLIND_PATH, [{"id": r["id"], "title": r["title"], "label": "", "hard": False} for r in sample])
    write_jsonl(
        MODEL_PATH,
        [{k: r[k] for k in ("id", "source", "published_at", "label", "confidence")} for r in sample],
    )

    print(f"pool size: {len(pool)} (published_at >= {start.isoformat()}, scored)")
    print(f"seed: {SEED}")
    print(f"rows written: {len(sample)}")
    print(f"blind: {BLIND_PATH}")
    print(f"model labels: {MODEL_PATH}")


if __name__ == "__main__":
    main()
