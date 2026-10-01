"""Mood index (T3.3): arithmetic, ET day assignment, hybrid recompute. In-memory store."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from itertools import count

import pytest

from newsmood.config import get_settings
from newsmood.data.feeds import Headline
from newsmood.data.store import connect, get_daily_index, upsert_headlines, write_scores
from newsmood.reporting.aggregate import compute_days, compute_mood

LOW = 0.60
FETCHED = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
_ids = count()


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def row(label: str, conf: float, source: str = "cnbc") -> dict:
    return {"id": f"h{next(_ids)}", "source": source, "label": label, "confidence": conf}


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def conn():
    c = connect(":memory:")
    yield c
    c.close()


def add(conn, published_at: datetime | None, label: str = "positive", conf: float = 0.9, source: str = "cnbc") -> str:
    """Ingest one headline and score it, through the real store functions."""
    n = next(_ids)
    h = Headline(
        id=f"id{n}", source=source, title=f"Headline {n}", title_norm=f"headline {n}",
        summary="", url=f"https://x.test/{n}", published_at=published_at, fetched_at=FETCHED,
    )
    upsert_headlines(conn, [h])
    write_scores(conn, [(h.id, label, conf)], scored_at=FETCHED)
    return h.id


# --- compute_mood: the pure arithmetic ---------------------------------------

def test_all_positive_is_plus_100():
    assert compute_mood([row("positive", c) for c in (0.9, 0.7, 0.99)], low_confidence=LOW).mood == 100


def test_all_neutral_is_zero():
    assert compute_mood([row("neutral", c) for c in (0.9, 0.7)], low_confidence=LOW).mood == 0


def test_empty_day_is_none_not_zero():
    m = compute_mood([], low_confidence=LOW)
    assert m.mood is None
    assert m.mean_confidence is None
    assert (m.n_headlines, m.n_low_conf, m.n_sources, m.terms) == (0, 0, 0, ())


def test_low_confidence_rows_still_produce_a_mood_and_are_counted():
    rows = [row("negative", 0.40), row("negative", 0.40), row("positive", 0.40)]
    m = compute_mood(rows, low_confidence=LOW)
    assert m.mood == pytest.approx(100 * (-0.4 - 0.4 + 0.4) / 1.2)
    assert m.n_low_conf == 3


def test_mixed_day_matches_hand_computed_value():
    rows = [
        row("positive", 0.9, "cnbc"), row("positive", 0.6, "yahoo"), row("neutral", 0.8, "cnbc"),
        row("negative", 0.5, "marketwatch"), row("negative", 0.7, "yahoo"),
    ]
    m = compute_mood(rows, low_confidence=LOW)
    # (0.9 + 0.6 - 0.5 - 0.7) / (0.9 + 0.6 + 0.8 + 0.5 + 0.7) = 0.3 / 3.5
    assert m.mood == pytest.approx(8.571428571428571)
    assert (m.n_positive, m.n_neutral, m.n_negative) == (2, 1, 2)
    assert m.mean_confidence == pytest.approx(0.7)
    assert m.n_low_conf == 1  # 0.5; 0.6 is not below 0.60
    assert m.source_counts == {"cnbc": 2, "yahoo": 2, "marketwatch": 1}
    assert m.n_sources == 3


def test_terms_itemize_the_mood():
    rows = [row("positive", 0.9), row("neutral", 0.8), row("negative", 0.55), row("negative", 0.97)]
    m = compute_mood(rows, low_confidence=LOW)
    assert [id_ for id_, _ in m.terms] == [r["id"] for r in rows]
    assert [t for _, t in m.terms] == pytest.approx([0.9, 0.0, -0.55, -0.97])
    total_conf = sum(r["confidence"] for r in rows)
    assert 100 * math.fsum(t for _, t in m.terms) / total_conf == pytest.approx(m.mood)


# --- day assignment in America/New_York ----------------------------------------

def _mood_rows(results):
    return {r.date: r.day.n_headlines for r in results}


def test_2330_et_lands_on_that_day_not_the_next(conn, settings):
    # 23:30 EDT on Sep 28 is 03:30 UTC on Sep 29.
    add(conn, utc("2026-09-28T23:30:00-04:00"))
    res = compute_days(conn, [date(2026, 9, 28), date(2026, 9, 29)], settings, now=FETCHED)
    assert _mood_rows(res) == {date(2026, 9, 28): 1, date(2026, 9, 29): 0}


def test_dst_fall_back_uses_est(conn, settings):
    # DST ends 2026-11-01 02:00 EDT. 03:30Z on Nov 2 is 22:30 EST (UTC-5) on
    # Nov 1. A fixed -04:00 offset would put it at 23:30 Nov 1 too, so the
    # second row (04:30Z = 23:30 EST Nov 1, 00:30 under a stale EDT offset)
    # is what separates zoneinfo from a hardcoded offset.
    add(conn, datetime(2026, 11, 2, 3, 30, tzinfo=timezone.utc))
    add(conn, datetime(2026, 11, 2, 4, 30, tzinfo=timezone.utc))
    now = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
    res = compute_days(conn, [date(2026, 11, 1), date(2026, 11, 2)], settings, now=now)
    assert _mood_rows(res) == {date(2026, 11, 1): 2, date(2026, 11, 2): 0}


def test_null_published_at_is_skipped(conn, settings):
    add(conn, None, label="negative")
    add(conn, utc("2026-09-30T10:00:00-04:00"), label="positive")
    [res] = compute_days(conn, [date(2026, 9, 30)], settings, now=FETCHED)
    assert res.day.n_headlines == 1
    assert res.day.mood == 100


# --- hybrid recompute ----------------------------------------------------------

NOW = datetime(2026, 9, 30, 20, 0, tzinfo=timezone.utc)  # 16:00 EDT, Sep 30
TODAY, YESTERDAY, OLDER = date(2026, 9, 30), date(2026, 9, 29), date(2026, 9, 27)


def at(day: date) -> datetime:
    return utc(f"{day.isoformat()}T12:00:00-04:00")


@pytest.mark.parametrize("day", [TODAY, YESTERDAY])
def test_today_and_yesterday_are_recomputed_when_rows_arrive(conn, settings, day):
    add(conn, at(day), label="positive")
    [first] = compute_days(conn, [day], settings, now=NOW)
    assert (first.action, first.day.mood) == ("recomputed", 100)

    late = add(conn, at(day), label="negative", conf=0.9)
    [second] = compute_days(conn, [day], settings, now=NOW)
    assert (second.action, second.day.mood, second.day.n_headlines) == ("recomputed", 0, 2)
    stored = get_daily_index(conn, day.isoformat())
    assert (stored["mood_index"], stored["n_headlines"]) == (0, 2)
    assert late in [id_ for id_, _ in stored["terms"]]


def test_older_day_with_no_row_is_written_once_then_never_rewritten(conn, settings):
    add(conn, at(OLDER), label="positive")
    [first] = compute_days(conn, [OLDER], settings, now=NOW)
    assert (first.action, first.day.mood) == ("written", 100)
    stored = get_daily_index(conn, OLDER.isoformat())

    add(conn, at(OLDER), label="negative")  # a late Yahoo item
    [second] = compute_days(conn, [OLDER], settings, now=NOW)
    assert second.action == "kept"
    assert get_daily_index(conn, OLDER.isoformat()) == stored
    assert second.day == first.day  # returned as stored, terms included


def test_empty_older_day_is_returned_but_not_stored(conn, settings):
    [res] = compute_days(conn, [OLDER], settings, now=NOW)
    assert (res.action, res.day.mood) == ("empty", None)
    assert get_daily_index(conn, OLDER.isoformat()) is None


def test_stored_row_records_model_and_stats(conn, settings):
    add(conn, at(TODAY), label="positive", conf=0.5, source="yahoo")
    add(conn, at(TODAY), label="neutral", conf=0.8, source="cnbc")
    compute_days(conn, [TODAY], settings, now=NOW)
    s = get_daily_index(conn, TODAY.isoformat())
    assert s["model_id"] == f"{settings.model.repo_id}@{settings.model.revision}"
    assert s["source_counts"] == {"cnbc": 1, "yahoo": 1}
    assert (s["n_positive"], s["n_neutral"], s["n_negative"], s["n_low_conf"], s["n_sources"]) == (1, 1, 0, 1, 2)
    assert len(s["computed_at"]) == 32


def test_unrequested_dates_never_produce_rows(conn, settings):
    add(conn, utc("2024-11-14T09:00:00-05:00"))  # stale Yahoo item
    add(conn, at(OLDER))
    add(conn, at(TODAY))
    compute_days(conn, [TODAY], settings, now=NOW)
    assert [r["date"] for r in conn.execute("SELECT date FROM daily_index")] == [TODAY.isoformat()]
