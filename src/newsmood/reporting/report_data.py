"""Day → finished report numbers (T4.2). Pure: no DB, no config file, no I/O.

T4.3 fetches the day's DayResult (reporting/aggregate.py) and its scored rows
and calls build_report_data(). template.py receives the result and does no
arithmetic, so every count, share, percent, band and ordering is decided here.

The mood is never recomputed: it comes from the DayResult, which for a frozen
day is the stored snapshot. Only rows whose ids are in its terms are shown,
so the pie, tables and mood describe the same headlines. Extra rows are
legitimate: a late headline can arrive for a frozen older day. They are
excluded and counted in n_late. A stored id with no row raises, because the
frozen mood would then rest on headlines the report can't show.

Rounding is one rule throughout: half-up on the decimal value (0.955 → 96%,
12.25% → 12.3%), via Decimal(repr(x)) so binary float noise can't push a
half-way value down.

Determinism: every list is ordered by confidence descending, then published_at
(earliest first, missing last), then id. Re-running a day gives the same
object, and so a byte-identical report.

Titles, urls and sources are passed through unescaped; escaping is template.py's job.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from newsmood.config import LedeBandsSettings, Settings
from newsmood.reporting.aggregate import DayResult

# Display order for the pie, recommended articles and the full list.
SENTIMENT_ORDER = ("negative", "positive", "neutral")
UNKNOWN_SOURCE = "unknown"
_LINK_SCHEMES = frozenset({"http", "https"})


@dataclass(frozen=True)
class Headline:
    id: str
    sentiment: str
    certainty_pct: int
    unsure: bool  # confidence < index.low_confidence
    title: str
    url: str | None  # as stored, untouched; None when missing
    linked: bool  # False → render title as plain text
    source: str


@dataclass(frozen=True)
class SentimentSlice:
    sentiment: str
    count: int
    share_pct: float  # 1 decimal, half-up; slices may sum to 99.9 or 100.1
    color: str  # report.colors[sentiment], so a color never follows position


@dataclass(frozen=True)
class RecommendedGroup:
    sentiment: str
    requested: int
    items: tuple[Headline, ...]

    @property
    def shortfall(self) -> bool:
        """True when fewer headlines exist than requested ("only 2 today")."""
        return len(self.items) < self.requested


@dataclass(frozen=True)
class SourceRow:
    source: str
    total: int
    negative: int
    neutral: int
    positive: int


@dataclass(frozen=True)
class ReportData:
    date: date  # calendar day in index.timezone
    mood: float | None  # 1 decimal, half-up; None on an empty day, never 0
    band: str | None  # "roughly flat", "mildly negative", ...; None on an empty day
    n_headlines: int
    n_sources: int
    n_unsure: int
    n_late: int  # rows for this date not in the frozen snapshot, excluded
    low_confidence_pct: int  # the threshold behind `unsure`, for the legend
    slices: tuple[SentimentSlice, ...]  # all three, in SENTIMENT_ORDER
    # Non-zero slices, count desc then SENTIMENT_ORDER. Mermaid colors slices
    # by position (current: declaration order; older: value-sorted). In this
    # order both agree, so pie1..pieN taken from these colors stay on their
    # sentiment.
    pie: tuple[SentimentSlice, ...]
    recommended: tuple[RecommendedGroup, ...]
    sources: tuple[SourceRow, ...]
    headlines: tuple[Headline, ...]
    model_id: str  # repo_id@revision, as stored
    gates: str  # T3.5 replaces this with a passed/total summary

    @property
    def is_empty(self) -> bool:
        return self.n_headlines == 0


def build_report_data(
    result: DayResult, rows: Iterable[Mapping[str, Any]], settings: Settings
) -> ReportData:
    """The finished, immutable numbers for one day's report.

    Each row needs id, label, confidence and title. source, url and
    published_at may be missing or None.
    """
    rows = list(rows)
    day = result.day
    rows, n_late = _snapshot_rows(result, rows)
    low = settings.index.low_confidence
    headlines = [_headline(r, low) for r in sorted(rows, key=_display_key)]
    mood = None if day.mood is None else _round(day.mood, "0.1")
    if mood == 0:
        mood = 0.0  # no "-0.0" in the header
    sources = _sources(headlines)

    counts = {"negative": day.n_negative, "positive": day.n_positive, "neutral": day.n_neutral}
    slices = (
        tuple(
            SentimentSlice(s, counts[s], _pct(counts[s], day.n_headlines), getattr(settings.report.colors, s))
            for s in SENTIMENT_ORDER
        )
        if day.n_headlines
        else ()
    )
    sample = settings.report.recommended
    recommended = (
        tuple(
            RecommendedGroup(s, getattr(sample, s), tuple(h for h in headlines if h.sentiment == s)[: getattr(sample, s)])
            for s in SENTIMENT_ORDER
        )
        if headlines
        else ()
    )
    return ReportData(
        date=result.date,
        mood=mood,
        band=band_label(mood, settings.lede_bands),
        n_headlines=day.n_headlines,
        n_sources=len(sources),
        n_unsure=sum(h.unsure for h in headlines),
        n_late=n_late,
        low_confidence_pct=_certainty_pct(low),
        slices=slices,
        pie=tuple(sorted((s for s in slices if s.count), key=lambda s: -s.count)),  # stable: ties keep SENTIMENT_ORDER
        recommended=recommended,
        sources=sources,
        headlines=tuple(headlines),
        model_id=result.model_id,
        gates="not yet implemented",
    )


def band_label(mood: float | None, bands: LedeBandsSettings) -> str | None:
    """Strength + direction for a mood, by |mood| with exclusive upper bounds:

        |mood| < flat      → "roughly flat"
        |mood| < mild      → "mildly <direction>"
        |mood| < moderate  → "moderately <direction>"
        otherwise          → "strongly <direction>"

    So exactly 5.0 is "mildly", 15.0 "moderately", 35.0 "strongly". Pass the
    displayed (rounded) mood, so the label always agrees with the printed
    number. None (an empty day) has no band.
    """
    if mood is None:
        return None
    size = abs(mood)
    if size < bands.flat:
        return "roughly flat"
    direction = "positive" if mood > 0 else "negative"
    strength = "mildly" if size < bands.mild else "moderately" if size < bands.moderate else "strongly"
    return f"{strength} {direction}"


def is_linkable(url: str | None) -> bool:
    """http(s) url with a host. Anything else renders as plain text.
    Nothing is fetched or checked; paywalls are deliberately not detected."""
    if not url:
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme.lower() in _LINK_SCHEMES and bool(parts.netloc)


def _snapshot_rows(result: DayResult, rows: list[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], int]:
    """The rows behind the day's stored terms, and how many later rows were left out."""
    term_ids = {id_ for id_, _ in result.day.terms}
    kept = [r for r in rows if r["id"] in term_ids]
    missing = term_ids - {r["id"] for r in kept}
    if missing:
        raise ValueError(f"{result.date}: {len(missing)} headlines in the index snapshot have no row, e.g. {min(missing)}")
    labels = Counter(r["label"] for r in kept)
    expected = {"negative": result.day.n_negative, "neutral": result.day.n_neutral, "positive": result.day.n_positive}
    if any(labels[s] != n for s, n in expected.items()):
        raise ValueError(f"{result.date}: row labels {dict(labels)} do not match the index counts {expected}")
    return kept, len(rows) - len(kept)


def _headline(r: Mapping[str, Any], low: float) -> Headline:
    url = r.get("url") or None
    source = (r.get("source") or "").strip() or UNKNOWN_SOURCE
    return Headline(
        id=r["id"],
        sentiment=r["label"],
        certainty_pct=_certainty_pct(r["confidence"]),
        unsure=r["confidence"] < low,
        title=r["title"],
        url=url,
        linked=is_linkable(url),
        source=source,
    )


def _display_key(r: Mapping[str, Any]):
    """Sentiment order, then confidence desc, published_at asc (missing last), id."""
    published = r.get("published_at")
    return (
        SENTIMENT_ORDER.index(r["label"]),
        -r["confidence"],
        published is None,
        "" if published is None else str(published),
        r["id"],
    )


def _sources(headlines: list[Headline]) -> tuple[SourceRow, ...]:
    per: dict[str, Counter] = {}
    for h in headlines:
        per.setdefault(h.source, Counter())[h.sentiment] += 1
    rows = [
        SourceRow(src, sum(c.values()), c["negative"], c["neutral"], c["positive"]) for src, c in per.items()
    ]
    return tuple(sorted(rows, key=lambda s: (-s.total, s.source)))


def _round(x: float, step: str) -> float:
    return float(Decimal(repr(x)).quantize(Decimal(step), rounding=ROUND_HALF_UP))


def _certainty_pct(conf: float) -> int:
    return int(Decimal(repr(conf)).scaleb(2).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _pct(count: int, total: int) -> float:
    return float((Decimal(100 * count) / Decimal(total)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
