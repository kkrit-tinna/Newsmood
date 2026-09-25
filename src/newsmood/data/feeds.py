"""RSS fetch + normalize.

Titles are kept exactly as feedparser returns them — no cleaning. They go to
DistilBERT unmodified (CLAUDE.md, "Raw text to the transformer").
"""

from __future__ import annotations

import calendar
import hashlib
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import feedparser

from newsmood.config import FeedSettings, IngestSettings


@dataclass(frozen=True)
class Headline:
    id: str
    source: str
    title: str
    summary: str
    url: str
    published_at: datetime | None
    fetched_at: datetime


@dataclass
class FeedResult:
    name: str
    headlines: list[Headline] = field(default_factory=list)
    skipped: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


Fetcher = Callable[[str, str, float], bytes]


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def fetch_url(url: str, user_agent: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_feed(body: bytes, source: str, fetched_at: datetime) -> tuple[list[Headline], int]:
    """Normalize one feed body. Returns (headlines, number of entries skipped).

    An entry without a title or link is skipped: no title means nothing to
    classify, no link means no stable id.
    """
    parsed = feedparser.parse(body)
    headlines: list[Headline] = []
    skipped = 0
    for entry in parsed.entries:
        title = entry.get("title")
        url = entry.get("link")
        if not title or not url:
            skipped += 1
            continue
        headlines.append(
            Headline(
                id=url_hash(url),
                source=source,
                title=title,
                summary=entry.get("summary", ""),
                url=url,
                published_at=_published_utc(entry),
                fetched_at=fetched_at,
            )
        )
    return headlines, skipped


def _published_utc(entry) -> datetime | None:
    # feedparser normalizes published_parsed to a UTC struct_time.
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed is None:
        return None
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)


def fetch_all(
    settings: IngestSettings,
    fetch: Fetcher = fetch_url,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> list[FeedResult]:
    """Fetch every configured feed sequentially, pausing between requests.

    A feed that errors is reported in its FeedResult rather than raised, so
    one dead feed does not stop the others.
    """
    results: list[FeedResult] = []
    for i, feed in enumerate(settings.feeds):
        if i > 0:
            sleep(settings.delay_seconds)
        results.append(_fetch_one(feed, settings, fetch, now))
    return results


def _fetch_one(feed: FeedSettings, settings: IngestSettings, fetch: Fetcher, now) -> FeedResult:
    fetched_at = now()
    try:
        body = fetch(feed.url, settings.user_agent, settings.timeout_seconds)
    except Exception as exc:  # network errors, HTTP errors, timeouts
        return FeedResult(name=feed.name, error=f"{type(exc).__name__}: {exc}")
    headlines, skipped = parse_feed(body, feed.name, fetched_at)
    if not headlines:
        return FeedResult(name=feed.name, skipped=skipped, error="no entries")
    return FeedResult(name=feed.name, headlines=headlines, skipped=skipped)


def format_dry_run(results: list[FeedResult]) -> str:
    width = max(len(r.name) for r in results)
    lines = []
    for r in results:
        if r.ok:
            line = f"{r.name:<{width}}  {len(r.headlines):>4} entries"
            if r.skipped:
                line += f"  ({r.skipped} skipped: no title or link)"
        else:
            line = f"{r.name:<{width}}  FAILED  {r.error}"
        lines.append(line)
    live = sum(r.ok for r in results)
    total = sum(len(r.headlines) for r in results)
    lines.append(f"{live}/{len(results)} feeds live, {total} entries total")
    return "\n".join(lines)
