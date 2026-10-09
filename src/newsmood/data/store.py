"""SQLite storage. The only module that imports sqlite3 (CLAUDE.md).

The schema is declared here and nowhere else. Downstream readers — scoring
(T2.5), the mood index (T3.3), gates (T3.5), explain queries (T4.2b) — rely
on the shape below, not on whatever an insert happened to create.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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

# One row per ET calendar day (T3.3), written only by reporting/aggregate.py.
# Per-label counts are columns: the label set is fixed by the headlines CHECK,
# and T3.5's gates and the trend query read them directly. source_counts and
# terms are JSON text: the source set follows config, and terms is one entry
# per headline, kept so a frozen day's top movers itemize its stored mood.
# mood_index is NULL for a day with no scored headlines, never 0.
DAILY_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_index (
    date             TEXT PRIMARY KEY,        -- YYYY-MM-DD, calendar day in index.timezone
    mood_index       REAL CHECK (mood_index BETWEEN -100.0 AND 100.0),
    n_headlines      INTEGER NOT NULL,
    n_positive       INTEGER NOT NULL,
    n_neutral        INTEGER NOT NULL,
    n_negative       INTEGER NOT NULL,
    mean_confidence  REAL,
    n_low_conf       INTEGER NOT NULL,
    n_sources        INTEGER NOT NULL,
    source_counts    TEXT NOT NULL,           -- JSON {source: count}
    terms            TEXT NOT NULL,           -- JSON [[headline id, conf * s], ...]
    model_id         TEXT NOT NULL,           -- repo_id@revision that produced the labels
    computed_at      TEXT NOT NULL,
    CHECK (n_headlines = n_positive + n_neutral + n_negative)
)
"""
_DAILY_JSON = ("source_counts", "terms")


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


def unscored(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """(id, title) for every row not yet scored, in a stable order. The raw
    title, never title_norm: that is what DistilBERT sees."""
    return [
        (row["id"], row["title"]) for row in conn.execute("SELECT id, title FROM headlines WHERE label IS NULL ORDER BY id")
    ]


def write_scores(conn: sqlite3.Connection, scores: Iterable[tuple[str, str, float]], scored_at: datetime) -> int:
    """Set label, confidence and scored_at on each (id, label, confidence), in
    one transaction. Only unscored rows are touched, so a score never
    overwrites an earlier one. Returns the number of rows updated."""
    ts = _ts(scored_at)
    rows = [(label, confidence, ts, id_) for id_, label, confidence in scores]
    with conn:
        before = conn.total_changes
        conn.executemany(
            "UPDATE headlines SET label = ?, confidence = ?, scored_at = ? WHERE id = ? AND label IS NULL", rows
        )
        return conn.total_changes - before


def scored_rows_between(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Scored rows with start <= published_at < end, as plain dicts.

    The bounds go through _ts, so this is a string range over the fixed-width
    UTC format (see _ts). Rows without a published_at have no day and are
    excluded explicitly; unscored rows are left for `newsmood score`.
    """
    sql = """
        SELECT id, source, label, confidence, published_at FROM headlines
        WHERE published_at IS NOT NULL AND label IS NOT NULL
          AND published_at >= ? AND published_at < ?
        ORDER BY published_at, id
    """
    return [dict(row) for row in conn.execute(sql, (_ts(start), _ts(end)))]


def scored_headlines_between(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[dict[str, Any]]:
    """The same rows as scored_rows_between, with title and url added: the
    full records a report shows (T4.3). Same filter and bounds, so a day's
    report rows and its index rows can't disagree on membership."""
    sql = """
        SELECT id, title, url, source, published_at, label, confidence FROM headlines
        WHERE published_at IS NOT NULL AND label IS NOT NULL
          AND published_at >= ? AND published_at < ?
        ORDER BY published_at, id
    """
    return [dict(row) for row in conn.execute(sql, (_ts(start), _ts(end)))]


def first_fetched_at(conn: sqlite3.Connection) -> datetime | None:
    """When the store's history begins: the earliest fetched_at, or None when
    empty. published_at can't say this, since feeds carry items years old."""
    value = conn.execute("SELECT min(fetched_at) FROM headlines").fetchone()[0]
    return None if value is None else datetime.fromisoformat(value)


def last_fetched_at(conn: sqlite3.Connection) -> datetime | None:
    """The newest fetched_at in the store, or None when empty. A row keeps its
    first-seen fetched_at, so this moves only when ingest stores something new
    (T3.5's pipeline_fresh gate)."""
    value = conn.execute("SELECT max(fetched_at) FROM headlines").fetchone()[0]
    return None if value is None else datetime.fromisoformat(value)


def unscored_between(conn: sqlite3.Connection, start: datetime, end: datetime) -> int:
    """How many rows with start <= published_at < end still have no label: the
    rows scored_rows_between would show once scored (T3.5's all_scored gate)."""
    sql = """
        SELECT count(*) FROM headlines
        WHERE published_at IS NOT NULL AND label IS NULL
          AND published_at >= ? AND published_at < ?
    """
    return conn.execute(sql, (_ts(start), _ts(end))).fetchone()[0]


def get_daily_index(conn: sqlite3.Connection, date: str) -> dict[str, Any] | None:
    """The stored row for one YYYY-MM-DD, JSON columns decoded, or None."""
    row = conn.execute("SELECT * FROM daily_index WHERE date = ?", (date,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    for col in _DAILY_JSON:
        out[col] = json.loads(out[col])
    return out


def upsert_daily_index(conn: sqlite3.Connection, row: dict[str, Any], computed_at: datetime) -> None:
    """Insert or replace one day's row. Whether a day may be rewritten is the
    caller's rule (aggregate.compute_days), not the store's."""
    values = dict(row, computed_at=_ts(computed_at))
    for col in _DAILY_JSON:
        values[col] = json.dumps(values[col], sort_keys=col == "source_counts")
    cols = list(values)
    sql = (
        f"INSERT INTO daily_index ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
        f"ON CONFLICT (date) DO UPDATE SET {', '.join(f'{c} = excluded.{c}' for c in cols if c != 'date')}"
    )
    with conn:
        conn.execute(sql, [values[c] for c in cols])


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
    conn.execute(DAILY_INDEX_SCHEMA)
    conn.commit()
