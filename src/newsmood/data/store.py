"""SQLite storage. The only module that imports sqlite3 (CLAUDE.md).

The schema is declared here and nowhere else. Downstream readers — scoring
(T2.5), the mood index (T3.3), gates (T3.5), explain queries (T4.2b) — rely
on the shape below, not on whatever an insert happened to create.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from newsmood.data.feeds import Headline

# Timestamps are ISO-8601 UTC strings. label/confidence/scored_at stay NULL
# until T2.5 scores the row, and are set together or not at all.
HEADLINES_SCHEMA = """
CREATE TABLE IF NOT EXISTS headlines (
    id            TEXT PRIMARY KEY,           -- sha256 of url, from feeds.url_hash
    source        TEXT NOT NULL,
    title         TEXT NOT NULL,              -- raw; what DistilBERT sees
    title_norm    TEXT NOT NULL UNIQUE,       -- dedupe key only; never sent to the model
    summary       TEXT NOT NULL DEFAULT '',
    url           TEXT NOT NULL,
    published_at  TEXT,                       -- NULL when the feed gives no date
    fetched_at    TEXT NOT NULL,
    label         TEXT CHECK (label IN ('positive', 'neutral', 'negative')),
    confidence    REAL CHECK (confidence BETWEEN 0.0 AND 1.0),
    scored_at     TEXT,
    CHECK (
        (label IS NULL AND confidence IS NULL AND scored_at IS NULL)
        OR (label IS NOT NULL AND confidence IS NOT NULL AND scored_at IS NOT NULL)
    )
)
"""


# Two unique keys, two conflict paths, one rule for both: the first-seen row
# keeps its id, source, title, url, summary and fetched_at. Only published_at
# may move, and only earlier. Scoring columns are never touched by ingest.
#   title_norm conflict: the same story at another outlet, or the same
#                        article under a changed tracking parameter.
#   id conflict:         the same url with an edited headline.
# SQLite's min() returns NULL if either side is NULL; coalesce picks the
# non-null side, and the first-seen value when both are NULL.
_EARLIEST_PUBLISHED = (
    "published_at = coalesce(min(headlines.published_at, excluded.published_at), "
    "headlines.published_at, excluded.published_at)"
)
_UPSERT = f"""
INSERT INTO headlines (id, source, title, title_norm, summary, url, published_at, fetched_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (title_norm) DO UPDATE SET {_EARLIEST_PUBLISHED}
ON CONFLICT (id) DO UPDATE SET {_EARLIEST_PUBLISHED}
"""


@dataclass(frozen=True)
class UpsertResult:
    received: int
    inserted: int

    @property
    def duplicates(self) -> int:
        return self.received - self.inserted

    def summary(self) -> str:
        return f"{self.inserted} inserted, {self.duplicates} duplicates of {self.received} received"


def _ts(dt: datetime | None) -> str | None:
    """Serialize a timestamp for storage. Every stored timestamp goes through here.

    Format: ``YYYY-MM-DDTHH:MM:SS.ffffff+00:00`` — ISO-8601, always UTC,
    always microseconds, always the ``+00:00`` suffix. 32 characters, fixed.

    INVARIANT: _UPSERT's earliest-published_at rule compares these with
    SQLite's min() on TEXT, i.e. as strings. That orders correctly only
    because every value has this exact width and offset. Storing local
    offsets, dropping microseconds, or using a "Z" suffix for some rows would
    make min() pick the wrong timestamp without raising anything.
    """
    return None if dt is None else dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def upsert_headlines(conn: sqlite3.Connection, headlines: Iterable[Headline]) -> UpsertResult:
    """Insert headlines, deduping on title_norm. Re-running is a no-op.

    Timestamps are written via _ts and compared as strings in _UPSERT; see
    _ts for the fixed-width UTC format that comparison depends on.

    Order matters: within one call, the first headline seen for a title_norm
    is the one kept, so feed order in config decides which source wins.
    """
    before = _count(conn)
    rows = [
        (h.id, h.source, h.title, h.title_norm, h.summary, h.url, _ts(h.published_at), _ts(h.fetched_at))
        for h in headlines
    ]
    with conn:
        conn.executemany(_UPSERT, rows)
    return UpsertResult(received=len(rows), inserted=_count(conn) - before)


def _count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT count(*) FROM headlines").fetchone()[0]


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the database, creating the file and schema if absent."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(HEADLINES_SCHEMA)
    conn.commit()
