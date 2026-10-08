"""The computed "Why" section (T4.2b). No DB, no network, no API key.

Three blocks: 0, 1 and 3. Block 1 (contribution %) lives in report_data.py,
beside the Recommended articles it annotates. Block 2 (distinctive terms) was
dropped (IMPLEMENTATION_GUIDE.md §9, 2026-10-07); its lift function survives
here only to name block 3's clusters.

Block 0, the lede, is one sentence assembled from slots, each dropped when
its data can't support it:

    direction + strength   always; lede_bands on the displayed 1-decimal mood,
                           the same rule as the header (report_data.band_label)
    change                 delta vs the previous session, on displayed moods;
                           dropped when |delta| < lede.change_min_delta or no
                           previous session qualifies
    driver                 the cluster (block 3) carrying the largest share of
                           tilt in the mood's direction. "with no dominant
                           theme" when clustering ran and nothing qualified
                           (or the band is "roughly flat"); left out entirely
                           when the day is below explain.min_headlines, since
                           nothing was clustered and nothing can be claimed

The previous session is the most recent day in D-1 .. D-N (N =
lede.prev_session_lookback_days, inclusive) that has a stored daily_index row
with at least lede.prev_session_min_headlines. Days without an ingest run hold
1-4 headlines, so the floor skips them, weekends or not. Stored rows come
through an injected lookup, so this module never touches the store; the
pipeline's lookup computes a missing candidate day with the index code.

Block 3, clusters: agglomerative, cosine distance, complete linkage, on all
of today's rows. Cosine compares which words two headlines share, regardless
of length. Complete linkage merges two groups only if their farthest pair is
within the threshold, so no headline joins a theme by being close to just one
member (single linkage chains A~B~C into one group). A cluster's share is its
Σ(conf·s) over the day's Σ|conf·s|: block 1's denominator, bounded ±100%.
A cluster is named from its own rows, never the day's: terms in at least
explain.min_term_headlines of its members, ranked by lift = mean TF-IDF over
the members / mean over the background (D-30 .. D-1). While stored history is
shorter than explain.background_days, the background is too thin for lift, so
names rank by raw count instead.

The vectorizer (explain.vectorizer):
  window      TfidfVectorizer fit per report run on today plus the background
              rows, never persisted. fit learns the vocabulary and IDF;
              transform maps text onto that fixed vocabulary and silently
              drops unseen words, so today has to be in the fit.
  phrasebank  the T1.4 vectorizer from logreg.artifact_dir (dev-only: models/
              is gitignored). The one file this module reads. Its OOV rate
              is printed in the report; above explain.oov_warn the pipeline
              suggests window on stderr.
Text goes through clean_for_tfidf, never to the transformer.

Determinism: rows are put in the report's canonical order (confidence desc,
published_at, id) before vectorizing, every ranking has an explicit
tie-break, and the phrase bank is chosen by a SHA-256 digest of the date,
never hash(), which is salted per process.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from newsmood.config import Settings
from newsmood.preprocessing import clean_for_tfidf
from newsmood.reporting.aggregate import DayMood
from newsmood.reporting.report_data import ReportData, round_half_up

# Letter first, 2+ characters: bare numbers like "721" survive cleaning, and
# curly apostrophes split "I’m" / "don’t" into lone "m" / "t".
TOKEN_PATTERN = r"(?u)\b[^\W\d_][\w-]+\b"
PHRASEBANK_FILE = "tfidf_vectorizer.joblib"

# A day's stored index row, or None when the day has none (an empty day is
# never stored). The pipeline computes a missing candidate before answering.
SessionLookup = Callable[[date], DayMood | None]

# Slot order in the date digest: one byte per slot, so slots vary independently.
_SUBJECT, _CHANGE, _COUNT, _DRIVER = range(4)

SUBJECTS = ("Financial news", "Market news", "Financial coverage")
COUNTS = ("across {n}", "over {n}", "from {n}")
# {up}/{down} pairs, so a variant reads right in either direction.
CHANGES = (
    ("up {d} from {when}", "down {d} from {when}"),
    ("{d} higher than {when}", "{d} lower than {when}"),
    ("a rise of {d} from {when}", "a drop of {d} from {when}"),
)
DRIVERS = (
    "driven mainly by {n} on {terms}",
    "led by {n} on {terms}",
    "with {n} on {terms} doing most of the work",
)
NO_DRIVERS = ("with no dominant theme", "with no single theme in the lead")

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True)
class PreviousSession:
    date: date
    mood: float  # displayed: 1 decimal, half-up
    n_headlines: int


@dataclass(frozen=True)
class Driver:
    """The theme the lede names. Built from block 3 (T4.2b part B)."""

    n_headlines: int
    terms: tuple[str, ...]


class ExplainError(RuntimeError):
    """A configuration the Why section can't run with."""


@dataclass(frozen=True)
class ThemeInputs:
    """What blocks 2-3 read, gathered by the pipeline."""

    rows: Sequence[Mapping[str, Any]]  # the day's scored rows; ones outside the snapshot are ignored
    terms: Sequence[tuple[str, float]]  # the snapshot's (id, conf·s)
    background: Sequence[Mapping[str, Any]]  # stored rows in D-background_days .. D-1
    history_days: int  # days from the store's first fetch to D


@dataclass(frozen=True)
class TermScore:
    term: str
    lift: float | None  # inf when absent from the background; None when ranked by count
    count: int  # rows containing the term


@dataclass(frozen=True)
class Cluster:
    name: tuple[str, ...]  # top terms of the cluster's own rows
    ids: tuple[str, ...]  # members, canonical order
    titles: tuple[str, ...]
    n_negative: int
    n_neutral: int
    n_positive: int
    share_pct: float  # 100 × Σ(conf·s) / day Σ|conf·s|, signed, 1 decimal

    @property
    def size(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class Oov:
    vectorizer: str  # "window" or "phrasebank"
    rate_pct: float  # unigram tokens of today's headlines outside the vocabulary, 1 decimal
    n_oov: int
    n_tokens: int
    suggest_window: bool  # phrasebank mode above explain.oov_warn: stderr advice, not report text


@dataclass(frozen=True)
class WindowFit:
    """What window mode's vectorizer was fit on, for the report's vocabulary line."""

    n_headlines: int
    first: date  # earliest ET day among the fit rows
    last: date  # the reported day


@dataclass(frozen=True)
class Explanation:
    lede: str
    previous: PreviousSession | None
    change: float | None  # displayed delta, None when the slot is dropped
    driver: Driver | None
    clustered: bool  # day reached explain.min_headlines, so block 3 ran
    n_headlines: int
    min_headlines: int
    clusters: tuple[Cluster, ...]  # shown clusters only; empty → "no clear theme today"
    oov: Oov | None  # phrasebank mode only
    fit: WindowFit | None  # window mode only


def build_explanation(
    data: ReportData, settings: Settings, *, lookup: SessionLookup, themes: ThemeInputs
) -> Explanation | None:
    """The Why section for one day's finished report numbers. None on an
    empty day, which has no mood to explain."""
    if data.is_empty:
        return None
    cfg = settings.explain
    previous = find_previous_session(data.date, lookup, settings)
    change = mood_change(data.mood, previous, settings)

    snapshot = dict(themes.terms)
    rows = sorted((r for r in themes.rows if r["id"] in snapshot), key=canonical_key)
    texts = [clean_for_tfidf(r["title"]) for r in rows]
    background = [clean_for_tfidf(r["title"]) for r in themes.background]
    vectorizer = build_vectorizer(settings, texts + background)
    if cfg.vectorizer == "phrasebank":
        oov, fit = oov_rate(vectorizer, texts, settings), None
    else:
        oov, fit = None, window_fit(rows, themes.background, data.date, settings)

    clustered = data.n_headlines >= cfg.min_headlines
    clusters = ()
    if clustered:
        by_count = themes.history_days < cfg.background_days
        clusters = find_clusters(rows, texts, background, vectorizer, snapshot, settings, by_count=by_count)
    driver = find_driver(data, clusters, settings)
    return Explanation(
        lede=build_lede(data, previous=previous, change=change, driver=driver, clustered=clustered),
        previous=previous,
        change=change,
        driver=driver,
        clustered=clustered,
        n_headlines=data.n_headlines,
        min_headlines=cfg.min_headlines,
        clusters=clusters,
        oov=oov,
        fit=fit,
    )


def canonical_key(r: Mapping[str, Any]):
    """The report's order: confidence desc, published_at asc (missing last), id."""
    published = r.get("published_at")
    return (-r["confidence"], published is None, "" if published is None else str(published), r["id"])


# --- vectorizer ---


def build_vectorizer(settings: Settings, fit_texts: Sequence[str]) -> TfidfVectorizer | None:
    """The vocabulary for blocks 2-3. window: fit here on today + background
    (today first). phrasebank: the T1.4 file. None when window's vocabulary
    comes out empty (too few repeated words), and every block then declines."""
    cfg = settings.explain
    if cfg.vectorizer == "phrasebank":
        return load_phrasebank_vectorizer(settings)
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(cfg.window_ngram_range),
        min_df=cfg.window_min_df,
        stop_words="english",
        token_pattern=TOKEN_PATTERN,
    )
    try:
        return vectorizer.fit(fit_texts)
    except ValueError:  # "empty vocabulary": no word reaches min_df
        return None


def load_phrasebank_vectorizer(settings: Settings) -> TfidfVectorizer:
    path = Path(settings.logreg.artifact_dir) / PHRASEBANK_FILE
    if not path.exists():
        raise ExplainError(
            f"explain.vectorizer is phrasebank, but {path} does not exist. It is the T1.4 baseline's "
            "vectorizer, dev-only and gitignored; run `newsmood eval` to fit it, or set "
            "explain.vectorizer: window (the default)."
        )
    return joblib.load(path)


def window_fit(
    rows: Sequence[Mapping[str, Any]], background: Sequence[Mapping[str, Any]], day: date, settings: Settings
) -> WindowFit:
    tz = ZoneInfo(settings.index.timezone)
    days = [datetime.fromisoformat(str(r["published_at"])).astimezone(tz).date() for r in [*rows, *background]]
    return WindowFit(n_headlines=len(rows) + len(background), first=min(days, default=day), last=day)


def oov_rate(vectorizer: TfidfVectorizer, texts: Sequence[str], settings: Settings) -> Oov:
    """Share of today's unigram tokens (after the vectorizer's own analyzer,
    so stop words are already gone in window mode) outside its vocabulary."""
    analyze, vocab = vectorizer.build_analyzer(), vectorizer.vocabulary_
    tokens = [t for text in texts for t in analyze(text) if " " not in t]
    n_oov = sum(t not in vocab for t in tokens)
    rate = n_oov / len(tokens) if tokens else 0.0
    mode = settings.explain.vectorizer
    return Oov(
        vectorizer=mode,
        rate_pct=round_half_up(100 * rate, "0.1"),
        n_oov=n_oov,
        n_tokens=len(tokens),
        suggest_window=mode == "phrasebank" and rate > settings.explain.oov_warn,
    )


# --- cluster names ---


def rank_terms(
    target: Sequence[str],
    background: Sequence[str],
    vectorizer: TfidfVectorizer | None,
    settings: Settings,
    *,
    by_count: bool,
) -> list[TermScore]:
    """Terms of `target` (cleaned texts) in at least explain.min_term_headlines
    of them, best first. Takes any set of rows; block 3 passes one cluster.

    Lift: mean TF-IDF over the target rows / mean over the background rows,
    zeros included. A term absent from the background has infinite lift (new
    this window). Order: lift desc, then term.
    By count (short history): rows containing the term, desc, then term.
    """
    if vectorizer is None or not target:
        return []
    X = vectorizer.transform(target)
    names = vectorizer.get_feature_names_out()
    counts = np.asarray((X > 0).sum(axis=0)).ravel()
    keep = np.flatnonzero(counts >= settings.explain.min_term_headlines)
    if by_count:
        ranked = sorted(keep, key=lambda j: (-counts[j], names[j]))
        return [TermScore(str(names[j]), None, int(counts[j])) for j in ranked]
    mean_t = np.asarray(X.mean(axis=0)).ravel()
    mean_b = np.asarray(vectorizer.transform(background).mean(axis=0)).ravel() if background else np.zeros_like(mean_t)
    lifts = {j: (math.inf if mean_b[j] == 0 else float(mean_t[j] / mean_b[j])) for j in keep}
    ranked = sorted(keep, key=lambda j: (-lifts[j], names[j]))
    return [TermScore(str(names[j]), lifts[j], int(counts[j])) for j in ranked]


# --- block 3: clusters ---


def find_clusters(
    rows: Sequence[Mapping[str, Any]],
    texts: Sequence[str],
    background: Sequence[str],
    vectorizer: TfidfVectorizer | None,
    snapshot: Mapping[str, float],
    settings: Settings,
    *,
    by_count: bool,
) -> tuple[Cluster, ...]:
    """Clusters of at least explain.min_cluster_size among `rows` (canonical
    order), ordered by |share| desc, size desc, then earliest member.

    A row whose vector is all zeros (every word outside the vocabulary) has
    no cosine distance to anything and is left out."""
    cfg = settings.explain
    if vectorizer is None:
        return ()
    X = vectorizer.transform(texts).toarray()
    usable = [i for i in range(len(rows)) if X[i].any()]
    if len(usable) < max(2, cfg.min_cluster_size):
        return ()
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=cfg.cluster_distance, metric="cosine", linkage="complete"
    ).fit(X[usable]).labels_
    groups: dict[int, list[int]] = {}
    for i, label in zip(usable, labels):
        groups.setdefault(int(label), []).append(i)  # labels are arbitrary; members stay in canonical order
    movement = math.fsum(abs(t) for t in snapshot.values())
    clusters = []
    for members in groups.values():
        if len(members) < cfg.min_cluster_size:
            continue
        member_rows = [rows[i] for i in members]
        tilt = math.fsum(snapshot[r["id"]] for r in member_rows)
        share = round_half_up(100 * tilt / movement, "0.1") if movement else 0.0
        name = rank_terms([texts[i] for i in members], background, vectorizer, settings, by_count=by_count)
        labels_ = [r["label"] for r in member_rows]
        clusters.append(
            (
                members[0],
                Cluster(
                    name=tuple(t.term for t in name[: cfg.cluster_name_terms]),
                    ids=tuple(r["id"] for r in member_rows),
                    titles=tuple(r["title"] for r in member_rows),
                    n_negative=labels_.count("negative"),
                    n_neutral=labels_.count("neutral"),
                    n_positive=labels_.count("positive"),
                    share_pct=share or 0.0,  # no "-0.0"
                ),
            )
        )
    clusters.sort(key=lambda fc: (-abs(fc[1].share_pct), -fc[1].size, fc[0]))
    return tuple(c for _, c in clusters)


def find_previous_session(day: date, lookup: SessionLookup, settings: Settings) -> PreviousSession | None:
    """Most recent day in D-1 .. D-N with a stored row and enough headlines.
    Candidates are asked for newest first, and the search stops at the first
    that qualifies, so older days are only computed when needed."""
    cfg = settings.lede
    for back in range(1, cfg.prev_session_lookback_days + 1):
        candidate = day - timedelta(days=back)
        stored = lookup(candidate)
        if stored is None or stored.mood is None or stored.n_headlines < cfg.prev_session_min_headlines:
            continue
        return PreviousSession(candidate, displayed_mood(stored.mood), stored.n_headlines)
    return None


def mood_change(mood: float | None, previous: PreviousSession | None, settings: Settings) -> float | None:
    """Displayed mood minus the previous session's displayed mood, or None
    when it is below lede.change_min_delta. Both are 1-decimal values, so the
    Decimal difference is exact: 5.0 is never 4.999…"""
    if mood is None or previous is None:
        return None
    delta = float(Decimal(repr(mood)) - Decimal(repr(previous.mood)))
    return delta if abs(delta) >= settings.lede.change_min_delta else None


def find_driver(data: ReportData, clusters: Sequence[Cluster], settings: Settings) -> Driver | None:
    """The cluster with the largest share of tilt in the mood's direction,
    among those of at least explain.min_cluster_size whose dominant sentiment
    matches the mood's sign in at least explain.cluster_purity of members.
    Never the largest by size: a cluster of neutral columns moves the index
    by 0. None on a "roughly flat" day, which has no direction to drive.
    Ties: larger cluster, then the order clusters arrive in."""
    if data.mood is None or data.band == "roughly flat":
        return None
    sign = 1 if data.mood > 0 else -1
    side = "positive" if sign > 0 else "negative"
    cfg = settings.explain
    qualifying = [
        (k, c)
        for k, c in enumerate(clusters)
        if c.size >= cfg.min_cluster_size
        and getattr(c, f"n_{side}") / c.size >= cfg.cluster_purity
        and c.share_pct * sign > 0
        and c.name
    ]
    if not qualifying:
        return None
    _, best = min(qualifying, key=lambda kc: (-kc[1].share_pct * sign, -kc[1].size, kc[0]))
    return Driver(best.size, best.name)


def build_lede(
    data: ReportData,
    *,
    previous: PreviousSession | None,
    change: float | None,
    driver: Driver | None,
    clustered: bool,
) -> str:
    """One sentence. Slots with no data are left out. On a day too thin to
    cluster the theme clause is left out too: "Financial news leaned
    moderately negative today (index -26.4) across 5 headlines." """
    pick = _picker(data.date)
    flat = data.band == "roughly flat"
    verb = f"was {data.band}" if flat else f"leaned {data.band}"
    index = f"index {data.mood:+.1f}"
    if change is not None and previous is not None:
        up, down = pick(_CHANGE, CHANGES)
        index += ", " + (up if change > 0 else down).format(d=f"{abs(change):.1f}", when=_when(previous.date, data.date))
    count = pick(_COUNT, COUNTS).format(n=_plural(data.n_headlines, "headline"))
    sentence = f"{pick(_SUBJECT, SUBJECTS)} {verb} today ({index}) {count}"
    if driver is not None and not flat:
        sentence += ", " + pick(_DRIVER, DRIVERS).format(n=_plural(driver.n_headlines, "headline"), terms=_join(driver.terms))
    elif clustered:
        sentence += ", " + pick(_DRIVER, NO_DRIVERS)
    return sentence + "."


def displayed_mood(mood: float) -> float:
    """The mood as the header prints it: 1 decimal, half-up, no -0.0."""
    return round_half_up(mood, "0.1") or 0.0


def _picker(day: date):
    digest = hashlib.sha256(day.isoformat().encode("ascii")).digest()

    def pick(slot: int, variants: tuple):
        return variants[digest[slot] % len(variants)]

    return pick


def _when(previous: date, day: date) -> str:
    if previous == day - timedelta(days=1):
        return "yesterday"
    return f"{_WEEKDAYS[previous.weekday()]} {_MONTHS[previous.month - 1]} {previous.day}"  # locale-free


def _join(terms: tuple[str, ...]) -> str:
    return terms[0] if len(terms) == 1 else ", ".join(terms[:-1]) + " and " + terms[-1]


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"
