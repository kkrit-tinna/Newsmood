"""`newsmood report` (T4.3): ingest → score → index → report file.

Offline only until T4.1. Nothing here reads ANTHROPIC_API_KEY or imports
reporting/claude.py. Gates (T3.5) are not wired in yet, so nothing blocks the
write: a thin or feedless day still produces a report that says so.

The feed fetcher, classifier, clock and output directory are injected, so
tests run with no network, no model and no real reports/ folder. Progress
and warnings go to stderr; the caller prints the returned path to stdout.

Every step is idempotent, so re-running a date is safe: ingest dedupes, score
touches only unscored rows, compute_days freezes older days, and the report
file is replaced atomically.
"""

from __future__ import annotations

import os
import re
import sys
from contextlib import closing
from dataclasses import dataclass
from functools import partial
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from newsmood.config import Settings
from newsmood.data import feeds, store
from newsmood.models import scoring
from newsmood.reporting.aggregate import DayResult, compute_days, day_bounds
from newsmood.reporting.report_data import build_report_data
from newsmood.reporting.template import render

ONLINE_NOT_IMPLEMENTED = "online mode not yet implemented (T4.1)"
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class ReportError(ValueError):
    """A request the pipeline refuses before doing any work."""


@dataclass(frozen=True)
class ReportRun:
    path: Path
    result: DayResult  # the reported day
    computed: tuple[DayResult, ...]  # every day compute_days touched this run


def resolve_date(requested: str | None, today: date) -> date:
    """The report date: `today` (ET) when none is given, else a strict
    YYYY-MM-DD that is not in the future."""
    if requested is None:
        return today
    if not _ISO_DATE.fullmatch(requested):
        raise ReportError(f"--date must be YYYY-MM-DD, got {requested!r}")
    try:
        day = date.fromisoformat(requested)
    except ValueError:
        raise ReportError(f"--date {requested!r} is not a calendar date") from None
    if day > today:
        raise ReportError(f"--date {requested} is in the future (today is {today.isoformat()} in ET)")
    return day


def days_to_compute(day: date, today: date) -> list[date]:
    """The reported day, plus yesterday when it is today: late headlines for
    yesterday still arrive, and the hybrid rule lets them update it."""
    return [day - timedelta(days=1), day] if day == today else [day]


def run_report(
    settings: Settings,
    *,
    date_arg: str | None = None,
    offline: bool = True,
    fetch: feeds.Fetcher = feeds.fetch_url,
    get_classifier: Callable[[], scoring.BatchClassifier] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    out_dir: Path | None = None,
) -> ReportRun:
    """Run the offline pipeline for one ET day and write {out_dir}/{date}.md.
    out_dir defaults to report.output_dir; tests pass a temp directory.

    Requests are validated before any I/O. The clock is read once, so "today"
    and every step in this run agree on it.
    """
    if not offline:
        raise ReportError(ONLINE_NOT_IMPLEMENTED)
    started = now()
    tz = ZoneInfo(settings.index.timezone)
    today = started.astimezone(tz).date()
    day = resolve_date(date_arg, today)
    if get_classifier is None:
        get_classifier = partial(_load_classifier, settings)

    with closing(store.connect(settings.store.db_path)) as conn:
        _ingest(conn, settings, fetch, started)
        n = scoring.score_store(conn, get_classifier, settings.model.batch_size)
        _progress(f"score: scored {n} headlines")
        computed = tuple(compute_days(conn, days_to_compute(day, today), settings, now=started))
        for r in computed:
            _progress(f"index: {r.date.isoformat()} {r.action} ({_mood(r)}, n={r.day.n_headlines})")
        result = next(r for r in computed if r.date == day)
        rows = store.scored_headlines_between(conn, *day_bounds(day, tz))

    out_dir = Path(settings.report.output_dir if out_dir is None else out_dir)
    path = write_atomic(out_dir / f"{day.isoformat()}.md", render(build_report_data(result, rows, settings)))
    return ReportRun(path=path, result=result, computed=computed)


def write_atomic(path: Path, text: str) -> Path:
    """Write to a temp file beside `path`, then rename over it. os.replace is
    atomic on one filesystem, so a crash leaves the old report or the new one,
    never half of either."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(text.encode("utf-8"))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _ingest(conn, settings: Settings, fetch: feeds.Fetcher, started: datetime) -> None:
    """Fetch and store. A failed feed is a warning, never an abort: with some
    feeds down the rest still count, and with all down the report is built
    from what is already stored. T3.5's gates decide later whether that is
    good enough to publish."""
    results = feeds.fetch_all(settings.ingest, fetch=fetch, now=lambda: started)
    for r in results:
        if not r.ok:
            _progress(f"warning: feed {r.name} failed: {r.error}")
    live = sum(r.ok for r in results)
    if results and live == 0:
        _progress(f"warning: all {len(results)} feeds failed; building the report from stored headlines only")
    upserted = store.upsert_headlines(conn, (h for r in results for h in r.headlines))
    _progress(f"ingest: {live}/{len(results)} feeds live, {upserted.summary()}")


def _load_classifier(settings: Settings) -> scoring.BatchClassifier:
    # Called by score_store only when something is unscored, so a run with
    # nothing new loads no model and downloads nothing.
    return scoring.load_batch_classifier(settings, scoring.resolve_device("auto"))


def _mood(r: DayResult) -> str:
    return "no mood" if r.day.mood is None else f"mood {r.day.mood:+.1f}"


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr)
