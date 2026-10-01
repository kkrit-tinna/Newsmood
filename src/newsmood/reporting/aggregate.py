"""Daily mood index (T3.3). The only implementation of the index (CLAUDE.md):
anything that evaluates or displays it imports from here.

    mood = 100 × Σ(confᵢ · sᵢ) / Σ(confᵢ),  s = +1 positive, 0 neutral, −1 negative

Every scored headline counts, weighted by its confidence. There is no cutoff:
index.low_confidence only feeds n_low_conf, so a reader can see how much of
the day rests on uncertain labels. A day with no scored headlines has no mood
(None), never 0, which would read as "perfectly balanced".

A day is a calendar day in index.timezone, assigned by published_at.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from newsmood.config import Settings
from newsmood.data import store

SIGN = {"positive": 1, "neutral": 0, "negative": -1}


@dataclass(frozen=True)
class DayMood:
    mood: float | None
    n_headlines: int
    n_positive: int
    n_neutral: int
    n_negative: int
    mean_confidence: float | None
    n_low_conf: int
    source_counts: dict[str, int]
    # (headline id, confᵢ · sᵢ) per headline. T4.2b's top movers itemize
    # these; Σ terms / Σ conf × 100 is the mood, so nothing recomputes it.
    terms: tuple[tuple[str, float], ...]

    @property
    def n_sources(self) -> int:
        return len(self.source_counts)


@dataclass(frozen=True)
class DayResult:
    date: date
    day: DayMood
    model_id: str
    # "recomputed": today/yesterday, upserted. "written": older day, first
    # write. "kept": older day already stored, returned as stored. "empty": no
    # scored headlines, nothing stored, so a later backfill can still fill it.
    action: str


def compute_mood(rows: Iterable[Mapping[str, Any]], *, low_confidence: float) -> DayMood:
    """The index and its stats summary for one day's scored rows. Pure.

    Each row needs id, source, label and confidence. Picking the day's rows
    is the caller's job (scored_rows_for_day).
    """
    rows = list(rows)
    labels = Counter(r["label"] for r in rows)
    terms = tuple((r["id"], r["confidence"] * SIGN[r["label"]]) for r in rows)
    total_conf = math.fsum(r["confidence"] for r in rows)
    return DayMood(
        mood=100 * math.fsum(t for _, t in terms) / total_conf if rows else None,
        n_headlines=len(rows),
        n_positive=labels["positive"],
        n_neutral=labels["neutral"],
        n_negative=labels["negative"],
        mean_confidence=total_conf / len(rows) if rows else None,
        n_low_conf=sum(r["confidence"] < low_confidence for r in rows),
        source_counts=dict(Counter(r["source"] for r in rows)),
        terms=terms,
    )


def day_bounds(day: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """[start, end) of one local calendar day, as UTC instants.

    Local midnight is attached to the zone and converted; zoneinfo looks up
    the offset in force on that date, so the window is 23 h on spring-forward
    day and 25 h on fall-back day. Midnight is never skipped or repeated in
    America/New_York (its switches happen at 02:00), so fold is moot here.
    """
    start = datetime.combine(day, time(0), tzinfo=tz)
    end = datetime.combine(day + timedelta(days=1), time(0), tzinfo=tz)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def scored_rows_for_day(conn, day: date, tz: ZoneInfo) -> list[dict[str, Any]]:
    return store.scored_rows_between(conn, *day_bounds(day, tz))


def model_id(settings: Settings) -> str:
    return f"{settings.model.repo_id}@{settings.model.revision}"


def compute_days(
    conn, days: Iterable[date], settings: Settings, *, now: datetime | None = None
) -> list[DayResult]:
    """Compute, store and return the index for exactly the requested days.

    Today and yesterday (in index.timezone, judged from `now`) are recomputed
    and upserted every call, since their headlines are still arriving. Older
    days are written once, if no row exists, and never rewritten after that.
    Days not requested are never touched.
    """
    tz = ZoneInfo(settings.index.timezone)
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(tz).date()
    mid = model_id(settings)
    results = []
    for day in days:
        recent = day >= today - timedelta(days=1)
        stored = None if recent else store.get_daily_index(conn, day.isoformat())
        if stored is not None:
            results.append(DayResult(day, _from_row(stored), stored["model_id"], "kept"))
            continue
        mood = compute_mood(scored_rows_for_day(conn, day, tz), low_confidence=settings.index.low_confidence)
        if mood.mood is None:
            results.append(DayResult(day, mood, mid, "empty"))
            continue
        store.upsert_daily_index(conn, _to_row(day, mood, mid), computed_at=now)
        results.append(DayResult(day, mood, mid, "recomputed" if recent else "written"))
    return results


def _to_row(day: date, m: DayMood, mid: str) -> dict[str, Any]:
    return {
        "date": day.isoformat(),
        "mood_index": m.mood,
        "n_headlines": m.n_headlines,
        "n_positive": m.n_positive,
        "n_neutral": m.n_neutral,
        "n_negative": m.n_negative,
        "mean_confidence": m.mean_confidence,
        "n_low_conf": m.n_low_conf,
        "n_sources": m.n_sources,
        "source_counts": m.source_counts,
        "terms": [list(t) for t in m.terms],
        "model_id": mid,
    }


def _from_row(row: Mapping[str, Any]) -> DayMood:
    return DayMood(
        mood=row["mood_index"],
        n_headlines=row["n_headlines"],
        n_positive=row["n_positive"],
        n_neutral=row["n_neutral"],
        n_negative=row["n_negative"],
        mean_confidence=row["mean_confidence"],
        n_low_conf=row["n_low_conf"],
        source_counts=dict(row["source_counts"]),
        terms=tuple((id_, term) for id_, term in row["terms"]),
    )
