"""RSS normalize + sequential fetch. No test touches the network."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from newsmood.config import FeedSettings, IngestSettings
from newsmood.data.feeds import fetch_all, format_dry_run, parse_feed

FETCHED_AT = datetime(2026, 9, 25, 1, 30, tzinfo=timezone.utc)

RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Test</title>
<item>
  <title>Stocks don't rally &amp; bonds slip</title>
  <link>https://example.com/a?mod=rss</link>
  <description>Summary A</description>
  <pubDate>Thu, 24 Sep 2026 17:12:04 -0400</pubDate>
</item>
<item>
  <title>No date here</title>
  <link>https://example.com/b</link>
</item>
<item>
  <title>No link here</title>
</item>
</channel></rss>"""


def _settings(*names: str) -> IngestSettings:
    return IngestSettings(
        user_agent="newsmood-test/0",
        timeout_seconds=7,
        delay_seconds=2,
        feeds=[FeedSettings(name=n, url=f"https://{n}.test/rss") for n in names],
    )


def test_parse_feed_normalizes_fields():
    headlines, skipped = parse_feed(RSS, "src", FETCHED_AT)
    assert skipped == 1
    a, b = headlines
    assert a.id == hashlib.sha256(b"https://example.com/a?mod=rss").hexdigest()
    assert a.source == "src"
    assert a.summary == "Summary A"
    assert a.fetched_at == FETCHED_AT
    assert a.published_at == datetime(2026, 9, 24, 21, 12, 4, tzinfo=timezone.utc)
    assert b.published_at is None
    assert b.summary == ""


def test_title_is_not_cleaned():
    # Raw text goes to the transformer: punctuation and contractions survive.
    (a, _), _ = parse_feed(RSS, "src", FETCHED_AT)
    assert a.title == "Stocks don't rally & bonds slip"


def test_fetch_all_is_sequential_polite_and_isolates_failures():
    calls, sleeps = [], []

    def fetch(url, user_agent, timeout):
        calls.append((url, user_agent, timeout))
        if "dead" in url:
            raise TimeoutError("timed out")
        return RSS

    results = fetch_all(_settings("one", "dead", "two"), fetch=fetch, sleep=sleeps.append, now=lambda: FETCHED_AT)

    assert [c[0] for c in calls] == ["https://one.test/rss", "https://dead.test/rss", "https://two.test/rss"]
    assert all(c[1] == "newsmood-test/0" and c[2] == 7 for c in calls)
    assert sleeps == [2, 2]  # between requests, not before the first
    assert [r.ok for r in results] == [True, False, True]
    assert len(results[0].headlines) == 2
    assert "TimeoutError" in results[1].error


def test_feed_with_no_entries_is_not_live():
    empty = b'<?xml version="1.0"?><rss version="2.0"><channel><title>x</title></channel></rss>'
    (result,) = fetch_all(_settings("empty"), fetch=lambda *a: empty, sleep=lambda s: None)
    assert not result.ok
    assert result.error == "no entries"


def test_format_dry_run_reports_counts_and_failures():
    results = fetch_all(
        _settings("one", "dead"),
        fetch=lambda url, *a: (_ for _ in ()).throw(OSError("refused")) if "dead" in url else RSS,
        sleep=lambda s: None,
    )
    out = format_dry_run(results)
    assert "one" in out and "2 entries" in out and "1 skipped" in out
    assert "dead" in out and "FAILED" in out
    assert out.splitlines()[-1] == "1/2 feeds live, 2 entries total"
