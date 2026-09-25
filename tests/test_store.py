"""SQLite store: schema, idempotent ingest, title_norm dedupe. In-memory only."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from newsmood.data.feeds import parse_feed
from newsmood.data.store import connect, init_schema, upsert_headlines

T1 = datetime(2026, 9, 25, 1, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 9, 25, 2, 0, tzinfo=timezone.utc)

# The same wire story at two outlets: different URL, different punctuation
# and casing, different pubDate. Live feeds had no such overlap on Sep 24–25,
# so the case is constructed here rather than assumed to work.
WIRE_A = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>A</title>
<item>
  <title>Fed Holds Rates Steady &#8212; Stocks Rise</title>
  <link>https://outlet-a.test/fed-holds</link>
  <pubDate>Thu, 24 Sep 2026 18:00:00 +0000</pubDate>
</item>
<item>
  <title>Oil slips on supply worries</title>
  <link>https://outlet-a.test/oil</link>
</item>
</channel></rss>"""

WIRE_B = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>B</title>
<item>
  <title>fed holds rates steady: stocks rise!</title>
  <link>https://outlet-b.test/markets/fed?mod=rss</link>
  <pubDate>Thu, 24 Sep 2026 17:30:00 +0000</pubDate>
</item>
<item>
  <title>Oil Slips on Supply Worries</title>
  <link>https://outlet-b.test/oil</link>
  <pubDate>Thu, 24 Sep 2026 16:00:00 +0000</pubDate>
</item>
<item>
  <title>Gold hits record</title>
  <link>https://outlet-b.test/gold</link>
</item>
</channel></rss>"""


@pytest.fixture
def conn():
    c = connect(":memory:")
    yield c
    c.close()


def _feeds():
    a, _ = parse_feed(WIRE_A, "outlet-a", T1)
    b, _ = parse_feed(WIRE_B, "outlet-b", T2)
    return a + b


def _rows(conn):
    return {r["title_norm"]: dict(r) for r in conn.execute("SELECT * FROM headlines")}


def test_schema_is_idempotent(conn):
    init_schema(conn)  # connect() already ran it once
    init_schema(conn)
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(headlines)")]
    assert cols == [
        "id", "source", "title", "title_norm", "summary", "url",
        "published_at", "fetched_at", "label", "confidence", "scored_at",
    ]


def test_ingesting_the_same_feed_twice_leaves_row_count_unchanged(conn):
    first = upsert_headlines(conn, _feeds())
    snapshot = _rows(conn)
    second = upsert_headlines(conn, _feeds())
    assert first.inserted == 3
    assert second.inserted == 0 and second.duplicates == 5
    assert _rows(conn) == snapshot


def test_cross_outlet_duplicate_collapses_to_first_seen(conn):
    result = upsert_headlines(conn, _feeds())
    assert result.received == 5 and result.inserted == 3 and result.duplicates == 2

    fed = _rows(conn)["fed holds rates steady stocks rise"]
    assert fed["source"] == "outlet-a"
    assert fed["url"] == "https://outlet-a.test/fed-holds"
    assert fed["title"] == "Fed Holds Rates Steady — Stocks Rise"  # raw, first-seen
    assert fed["fetched_at"] == T1.isoformat(timespec="microseconds")


def test_published_at_moves_to_earliest_non_null(conn):
    upsert_headlines(conn, _feeds())
    rows = _rows(conn)
    # Both dated: outlet-b's earlier time wins, source stays outlet-a.
    assert rows["fed holds rates steady stocks rise"]["published_at"].startswith("2026-09-24T17:30:00")
    # First-seen undated, duplicate dated: the date is filled in.
    assert rows["oil slips on supply worries"]["published_at"].startswith("2026-09-24T16:00:00")
    assert rows["oil slips on supply worries"]["source"] == "outlet-a"


def test_later_published_at_never_replaces_earlier(conn):
    (h,) = [h for h in _feeds() if h.source == "outlet-b" and "fed" in h.title_norm]
    upsert_headlines(conn, [h])
    upsert_headlines(conn, [replace(h, published_at=datetime(2026, 9, 24, 20, 0, tzinfo=timezone.utc))])
    upsert_headlines(conn, [replace(h, published_at=None)])
    (row,) = _rows(conn).values()
    assert row["published_at"].startswith("2026-09-24T17:30:00")


def test_fetched_at_never_moves(conn):
    (h, *_) = _feeds()
    upsert_headlines(conn, [h])
    upsert_headlines(conn, [replace(h, fetched_at=datetime(2026, 9, 30, tzinfo=timezone.utc))])
    (row,) = _rows(conn).values()
    assert row["fetched_at"] == T1.isoformat(timespec="microseconds")


def test_changed_tracking_parameter_keeps_original_id(conn):
    # Same article, new ?mod= value: new url hash, same title_norm.
    (h, *_) = _feeds()
    upsert_headlines(conn, [h])
    upsert_headlines(conn, [replace(h, id="different-hash", url=h.url + "?mod=new")])
    (row,) = _rows(conn).values()
    assert row["id"] == h.id and row["url"] == h.url


def test_edited_headline_at_same_url_keeps_first_seen_title(conn):
    (h, *_) = _feeds()
    upsert_headlines(conn, [h])
    result = upsert_headlines(conn, [replace(h, title="Fed Holds Rates", title_norm="fed holds rates")])
    assert result.inserted == 0
    (row,) = _rows(conn).values()
    assert row["title"] == h.title and row["title_norm"] == h.title_norm


def test_reingest_does_not_touch_scored_rows(conn):
    upsert_headlines(conn, _feeds())
    conn.execute(
        "UPDATE headlines SET label='positive', confidence=0.91, scored_at='2026-09-25T03:00:00.000000+00:00'"
        " WHERE title_norm='gold hits record'"
    )
    upsert_headlines(conn, _feeds())
    gold = _rows(conn)["gold hits record"]
    assert (gold["label"], gold["confidence"]) == ("positive", 0.91)


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE headlines SET label='positive'",  # half-scored
        "UPDATE headlines SET label='bullish', confidence=0.9, scored_at='x'",
        "UPDATE headlines SET label='positive', confidence=1.5, scored_at='x'",
    ],
)
def test_schema_rejects_invalid_scoring_state(conn, sql):
    upsert_headlines(conn, _feeds())
    with pytest.raises(Exception, match="CHECK constraint failed"):
        conn.execute(sql)


def test_stored_timestamps_are_fixed_width_utc(conn):
    # The earliest-published_at rule compares these as strings (see store._ts).
    # A non-UTC, whole-second input must still come out in the one format.
    (h, *_) = _feeds()
    est = timezone(timedelta(hours=-4))
    upsert_headlines(conn, [replace(h, published_at=datetime(2026, 9, 24, 13, 0, tzinfo=est))])
    (row,) = _rows(conn).values()
    assert row["published_at"] == "2026-09-24T17:00:00.000000+00:00"
    assert len(row["fetched_at"]) == 32 and row["fetched_at"].endswith("+00:00")
