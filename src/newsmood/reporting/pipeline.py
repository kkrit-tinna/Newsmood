"""`newsmood report` (T4.3): ingest → score → index → report file.

Offline only until T4.1. Nothing here reads ANTHROPIC_API_KEY or imports
reporting/claude.py.

Gates (T3.5, evaluation/gates.py) run twice. The blocks run after scoring and
before the index is computed: a blocked run writes quality/{date}.json, no
index row and no report, and raises GateBlocked. Checking before compute_days
matters for a past date, whose row is frozen once written. The disclosures
run after the index, on the snapshot the report shows, and land in the
report's data-notes line. quality/{date}.json is written on every run that
gets as far as the gates, pass or fail.

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
from dataclasses import dataclass, replace
from functools import partial
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from newsmood.config import Settings
from newsmood.data import feeds, store
from newsmood.evaluation import gates
from newsmood.models import scoring
from newsmood.reporting.aggregate import DayMood, DayResult, compute_days, day_bounds
from newsmood.reporting.explain import SessionLookup, ThemeInputs, build_explanation
from newsmood.reporting.report_data import build_report_data
from newsmood.reporting.template import render

ONLINE_NOT_IMPLEMENTED = "online mode not yet implemented (T4.1)"
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class ReportError(ValueError):
    """A request the pipeline refuses before doing any work."""


class GateBlocked(Exception):
    """A block gate failed. quality/{date}.json was written; the report was not."""

    def __init__(self, report: gates.GateReport, quality_path: Path, now: datetime):
        self.report = report
        self.quality_path = quality_path
        reasons = "; ".join(f"{r.name}: {gates.describe_failure(r, now)}" for r in report.failed)
        super().__init__(f"blocked by gate {reasons}. No report written; see {quality_path}")


@dataclass(frozen=True)
class ReportRun:
    path: Path
    quality_path: Path  # quality/{date}.json beside the reports
    gates: gates.GateReport
    result: DayResult  # the reported day
    computed: tuple[DayResult, ...]  # the reported day (and yesterday, when it is today)
    lookback: tuple[DayResult, ...]  # extra days computed to find the lede's previous session


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
    """Run the offline pipeline for one ET day and write {out_dir}/{date}.md
    and {out_dir}/quality/{date}.json. out_dir defaults to report.output_dir;
    tests pass a temp directory.

    Requests are validated before any I/O. The clock is read once, so "today"
    and every step in this run agree on it. Raises GateBlocked when a block
    gate fails.
    """
    if not offline:
        raise ReportError(ONLINE_NOT_IMPLEMENTED)
    started = now()
    tz = ZoneInfo(settings.index.timezone)
    today = started.astimezone(tz).date()
    day = resolve_date(date_arg, today)
    if get_classifier is None:
        get_classifier = partial(_load_classifier, settings)
    out_dir = Path(settings.report.output_dir if out_dir is None else out_dir)
    quality_path = out_dir / "quality" / f"{day.isoformat()}.json"

    with closing(store.connect(settings.store.db_path)) as conn:
        feed_status = _ingest(conn, settings, fetch, started)
        n = scoring.score_store(conn, get_classifier, settings.model.batch_size)
        _progress(f"score: scored {n} headlines")
        gate_inputs = gates.GateInputs(
            is_today=day == today,
            now=started,
            last_fetched_at=store.last_fetched_at(conn),
            n_unscored=store.unscored_between(conn, *day_bounds(day, tz)),
            feeds=feed_status,
            day=None,
        )
        blocks = gates.evaluate(gate_inputs, settings)
        if blocks.blocked:
            write_atomic(quality_path, blocks.to_json(day.isoformat(), started))
            raise GateBlocked(blocks, quality_path, started)
        computed = tuple(compute_days(conn, days_to_compute(day, today), settings, now=started))
        for r in computed:
            _progress(f"index: {r.date.isoformat()} {r.action} ({_mood(r)}, n={r.day.n_headlines})")
        result = next(r for r in computed if r.date == day)
        stats = gates.DayStats(result.day.n_headlines, result.day.source_counts, result.day.n_low_conf)
        report = gates.evaluate(replace(gate_inputs, day=stats), settings)
        write_atomic(quality_path, report.to_json(day.isoformat(), started))
        _progress(f"gates: {_gate_line(report)}")
        rows = store.scored_headlines_between(conn, *day_bounds(day, tz))
        data = build_report_data(result, rows, settings, gates=report.summary())
        lookup, lookback = session_lookup(conn, settings, started, computed)
        explanation = build_explanation(data, settings, lookup=lookup, themes=theme_inputs(conn, settings, result, rows))
        for r in lookback:
            _progress(f"index: {r.date.isoformat()} {r.action} for the previous session ({_mood(r)}, n={r.day.n_headlines})")
        if explanation is not None and explanation.oov is not None and explanation.oov.suggest_window:
            _progress(
                f"note: {explanation.oov.rate_pct:.1f}% of today's words are outside the phrasebank vocabulary "
                f"(explain.oov_warn {settings.explain.oov_warn:.0%}); consider explain.vectorizer: window"
            )

    path = write_atomic(out_dir / f"{day.isoformat()}.md", render(data, explanation))
    return ReportRun(
        path=path, quality_path=quality_path, gates=report, result=result, computed=computed, lookback=tuple(lookback)
    )


def theme_inputs(conn, settings: Settings, result: DayResult, rows: list[dict]) -> ThemeInputs:
    """Blocks 2-3's inputs: the day's rows and snapshot terms, the stored rows
    of the previous explain.background_days ET days, and how many days of
    history the store holds before the reported day (from its first fetch)."""
    tz = ZoneInfo(settings.index.timezone)
    day = result.date
    start = day_bounds(day - timedelta(days=settings.explain.background_days), tz)[0]
    background = store.scored_headlines_between(conn, start, day_bounds(day, tz)[0])
    first = store.first_fetched_at(conn)
    history_days = 0 if first is None else max(0, (day - first.astimezone(tz).date()).days)
    return ThemeInputs(rows=rows, terms=result.day.terms, background=background, history_days=history_days)


def session_lookup(
    conn, settings: Settings, started: datetime, computed: tuple[DayResult, ...]
) -> tuple[SessionLookup, list[DayResult]]:
    """The lede's view of stored daily_index rows. A day already computed this
    run is reused; any other is computed with the index's own rules (older
    days written once, empty days not stored) and appended to the returned
    list. An empty day answers None: it has no stored row."""
    known = {r.date: r for r in computed}
    extra: list[DayResult] = []

    def lookup(day: date) -> DayMood | None:
        if day not in known:
            (known[day],) = compute_days(conn, [day], settings, now=started)
            extra.append(known[day])
        r = known[day]
        return None if r.action == "empty" else r.day

    return lookup, extra


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


def _ingest(conn, settings: Settings, fetch: feeds.Fetcher, started: datetime) -> tuple[gates.FeedStatus, ...]:
    """Fetch and store, and return each feed's outcome for the gates. A failed
    feed is a warning here: with some feeds down the rest still count, and
    with all down the report is built from what is already stored, unless
    pipeline_fresh finds nothing fetched within gates.max_feed_staleness_hours."""
    results = feeds.fetch_all(settings.ingest, fetch=fetch, now=lambda: started)
    for r in results:
        if not r.ok:
            _progress(f"warning: feed {r.name} failed: {r.error}")
    live = sum(r.ok for r in results)
    if results and live == 0:
        _progress(f"warning: all {len(results)} feeds failed; building the report from stored headlines only")
    upserted = store.upsert_headlines(conn, (h for r in results for h in r.headlines))
    _progress(f"ingest: {live}/{len(results)} feeds live, {upserted.summary()}")
    return tuple(
        gates.FeedStatus(
            name=r.name,
            responded=r.ok,
            newest_published_at=max((h.published_at for h in r.headlines if h.published_at is not None), default=None),
        )
        for r in results
    )


def _load_classifier(settings: Settings) -> scoring.BatchClassifier:
    # Called by score_store only when something is unscored, so a run with
    # nothing new loads no model and downloads nothing.
    return scoring.load_batch_classifier(settings, scoring.resolve_device("auto"))


def _gate_line(report: gates.GateReport) -> str:
    return ", ".join(f"{r.name} {r.status}" for r in report.results)


def _mood(r: DayResult) -> str:
    return "no mood" if r.day.mood is None else f"mood {r.day.mood:+.1f}"


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr)
