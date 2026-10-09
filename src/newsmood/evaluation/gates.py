"""Quality gates (T3.5). Pure: no DB, no config file, no I/O.

Block broken runs, disclose thin days. A block that fails stops the run
before the index is written and before the report (or any future Claude
call); a disclosure never stops anything and adds a phrase to the report's
data-notes line. T3.3 already decided a thin day is shown with its n, not
hidden, so thinness is a disclosure, never a block.

    block     pipeline_fresh  today only: the newest stored fetched_at is
                              gates.max_feed_staleness_hours or more old
              all_scored      a row in the day's window is still unscored
    disclose  thin_day        n < explain.min_headlines
              feeds_down      fewer feeds responded than are configured
              single_source   one source's share of n > max_single_source_share
              low_confidence  low-confidence share of n > max_low_confidence_rate
              stale_feed      a responding feed's newest item is
                              max_feed_staleness_hours or more old

Status is "pass" or "fail" for a block, "pass" or "fired" for a disclosure,
and "not_applicable" when the gate has nothing to judge: a past --date run
(backfills read stored data by design, T4.3, so today's feeds say nothing
about them), an empty day, or day gates left unevaluated because a block
failed first (evaluating them needs the index, which a blocked run must not
write).

Staleness compares age >= threshold, so a threshold of 0 always fails. T4.4's
forced-failure check relies on that: a row this run just fetched is 0 h old.

Observed values hold timestamps, not ages, so quality/{date}.json is the same
on a re-run apart from run_at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Mapping

from newsmood.config import Settings

BLOCK, DISCLOSE = "block", "disclose"
PASS, FAIL, FIRED, NOT_APPLICABLE = "pass", "fail", "fired", "not_applicable"


@dataclass(frozen=True)
class FeedStatus:
    """One configured feed's outcome in this run's ingest."""

    name: str
    responded: bool
    newest_published_at: datetime | None  # newest dated item fetched; None if failed or undated


@dataclass(frozen=True)
class DayStats:
    """The reported day's index snapshot, as the report header shows it."""

    n_headlines: int
    source_counts: Mapping[str, int]
    n_low_conf: int


@dataclass(frozen=True)
class GateInputs:
    is_today: bool  # the run's date is today in index.timezone
    now: datetime  # the run's injected clock
    last_fetched_at: datetime | None  # newest fetched_at in the store; None when empty
    n_unscored: int  # rows in the day's window with no label after the score step
    feeds: tuple[FeedStatus, ...] | None  # None: ingest did not run this invocation
    day: DayStats | None  # None: not evaluated (a block failed first)


@dataclass(frozen=True)
class GateResult:
    name: str
    kind: str  # BLOCK | DISCLOSE
    status: str  # PASS | FAIL | FIRED | NOT_APPLICABLE
    observed: dict[str, Any] | None  # JSON-ready; None when not applicable
    threshold: float | None
    threshold_key: str | None  # the config key behind threshold

    def as_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "observed": self.observed,
            "status": self.status,
            "threshold": self.threshold,
            "threshold_key": self.threshold_key,
        }


@dataclass(frozen=True)
class GateSummary:
    """What the report footer and data-notes line show. Finished numbers:
    template.py only formats them."""

    n_passed: int  # blocks that passed
    n_applicable: int  # blocks that were judged
    n_not_applicable: int  # blocks that were not (pipeline_fresh on a past date)
    notes: tuple[GateResult, ...]  # fired disclosures, in gate order


@dataclass(frozen=True)
class GateReport:
    results: tuple[GateResult, ...]  # always all seven, in the order above

    @property
    def failed(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if r.status == FAIL)

    @property
    def blocked(self) -> bool:
        return bool(self.failed)

    def summary(self) -> GateSummary:
        blocks = [r for r in self.results if r.kind == BLOCK]
        judged = [r for r in blocks if r.status != NOT_APPLICABLE]
        return GateSummary(
            n_passed=sum(r.status == PASS for r in judged),
            n_applicable=len(judged),
            n_not_applicable=len(blocks) - len(judged),
            notes=tuple(r for r in self.results if r.status == FIRED),
        )

    def to_json(self, date: str, run_at: datetime) -> str:
        """quality/{date}.json. Sorted keys, gates in fixed order, so only
        run_at differs between re-runs over the same data."""
        doc = {
            "blocked": self.blocked,
            "date": date,
            "gates": [r.as_json() for r in self.results],
            "run_at": _iso(run_at),
        }
        return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def evaluate(inputs: GateInputs, settings: Settings) -> GateReport:
    g = settings.gates
    return GateReport(
        (
            _pipeline_fresh(inputs, g.max_feed_staleness_hours),
            _all_scored(inputs),
            _thin_day(inputs, settings.explain.min_headlines),
            _feeds_down(inputs),
            _single_source(inputs, g.max_single_source_share),
            _low_confidence(inputs, g.max_low_confidence_rate),
            _stale_feed(inputs, g.max_feed_staleness_hours),
        )
    )


def describe_failure(r: GateResult, now: datetime) -> str:
    """Why a block failed, for stderr, after its name. Ages are fine here:
    stderr is not compared between runs."""
    if r.name == "pipeline_fresh":
        newest = r.observed["newest_fetched_at"]
        if newest is None:
            return f"the store holds no fetched headlines ({r.threshold_key} {r.threshold:g})"
        age = (now - datetime.fromisoformat(newest)) / timedelta(hours=1)
        return f"newest fetch {newest} is {age:.1f} h old ({r.threshold_key} {r.threshold:g})"
    if r.name == "all_scored":
        return f"{r.observed['unscored']} headlines in the day's window are unscored after scoring"
    return str(r.observed)


def _pipeline_fresh(i: GateInputs, max_hours: float) -> GateResult:
    def result(status, observed):
        return GateResult("pipeline_fresh", BLOCK, status, observed, max_hours, "gates.max_feed_staleness_hours")

    if not i.is_today:
        return result(NOT_APPLICABLE, None)
    fresh = i.last_fetched_at is not None and not _stale(i.last_fetched_at, i.now, max_hours)
    return result(PASS if fresh else FAIL, {"newest_fetched_at": _iso(i.last_fetched_at)})


def _all_scored(i: GateInputs) -> GateResult:
    return GateResult(
        "all_scored", BLOCK, FAIL if i.n_unscored else PASS, {"unscored": i.n_unscored}, 0, None
    )


def _thin_day(i: GateInputs, min_headlines: int) -> GateResult:
    def result(status, observed):
        return GateResult("thin_day", DISCLOSE, status, observed, min_headlines, "explain.min_headlines")

    # An empty day already renders "No headlines for this date."
    if i.day is None or i.day.n_headlines == 0:
        return result(NOT_APPLICABLE, None)
    n = i.day.n_headlines
    return result(FIRED if n < min_headlines else PASS, {"headlines": n})


def _feeds_down(i: GateInputs) -> GateResult:
    if not i.is_today or i.feeds is None:
        return GateResult("feeds_down", DISCLOSE, NOT_APPLICABLE, None, None, None)
    failed = sorted(f.name for f in i.feeds if not f.responded)
    observed = {"configured": len(i.feeds), "failed": failed, "responded": len(i.feeds) - len(failed)}
    return GateResult("feeds_down", DISCLOSE, FIRED if failed else PASS, observed, None, None)


def _single_source(i: GateInputs, max_share: float) -> GateResult:
    def result(status, observed):
        return GateResult("single_source", DISCLOSE, status, observed, max_share, "gates.max_single_source_share")

    if i.day is None or i.day.n_headlines == 0:
        return result(NOT_APPLICABLE, None)
    n = i.day.n_headlines
    source, count = min(i.day.source_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    observed = {"count": count, "headlines": n, "share_pct": _pct(count, n), "source": source}
    return result(FIRED if count / n > max_share else PASS, observed)


def _low_confidence(i: GateInputs, max_rate: float) -> GateResult:
    def result(status, observed):
        return GateResult("low_confidence", DISCLOSE, status, observed, max_rate, "gates.max_low_confidence_rate")

    if i.day is None or i.day.n_headlines == 0:
        return result(NOT_APPLICABLE, None)
    n, low = i.day.n_headlines, i.day.n_low_conf
    return result(FIRED if low / n > max_rate else PASS, {"headlines": n, "unsure": low})


def _stale_feed(i: GateInputs, max_hours: float) -> GateResult:
    def result(status, observed):
        return GateResult("stale_feed", DISCLOSE, status, observed, max_hours, "gates.max_feed_staleness_hours")

    if not i.is_today or i.feeds is None:
        return result(NOT_APPLICABLE, None)
    # A responding feed with no dated item can't be judged and is left out.
    newest = {f.name: f.newest_published_at for f in i.feeds if f.responded and f.newest_published_at is not None}
    stale = sorted(name for name, ts in newest.items() if _stale(ts, i.now, max_hours))
    observed = {"newest_published_at": {name: _iso(ts) for name, ts in sorted(newest.items())}, "stale": stale}
    return result(FIRED if stale else PASS, observed)


def _stale(ts: datetime, now: datetime, max_hours: float) -> bool:
    return now - ts >= timedelta(hours=max_hours)


def _pct(count: int, total: int) -> int:
    """Whole percent, half-up: report_data.py's rounding rule."""
    return int((Decimal(100 * count) / Decimal(total)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.astimezone(timezone.utc).isoformat(timespec="microseconds")
