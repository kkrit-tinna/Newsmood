"""Computed Why section (T4.2b): plain objects, injected lookups. No DB, no
network, no model, no API."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, timedelta

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from newsmood.config import get_settings
from newsmood.reporting.aggregate import DayMood, DayResult, compute_mood
from newsmood.reporting.explain import (
    Driver,
    ExplainError,
    PreviousSession,
    ThemeInputs,
    build_explanation,
    build_lede,
    find_previous_session,
    mood_change,
)
from newsmood.reporting.report_data import ReportData, band_label, build_report_data
from newsmood.reporting.template import render

MID = "owner/model@" + "a" * 40
# Every slot picks its first variant on this date (SHA-256 of "2027-09-01", a Wednesday).
FIRST_VARIANTS_DAY = date(2027, 9, 1)
MON, SUN, SAT, FRI, THU = date(2026, 10, 5), date(2026, 10, 4), date(2026, 10, 3), date(2026, 10, 2), date(2026, 10, 1)


@pytest.fixture
def settings():
    return get_settings()


def report(settings, day=FIRST_VARIANTS_DAY, mood=-26.4, n=15) -> ReportData:
    return ReportData(
        date=day,
        mood=mood,
        band=band_label(mood, settings.lede_bands),
        n_headlines=n,
        n_sources=2,
        n_unsure=0,
        n_late=0,
        low_confidence_pct=60,
        slices=(),
        pie=(),
        recommended=(),
        sources=(),
        headlines=(),
        model_id=MID,
        gates="not yet implemented",
    )


def stored(mood, n) -> DayMood:
    return DayMood(mood, n, 0, n, 0, 0.8, 0, {"cnbc": n}, ())


NO_THEMES = ThemeInputs(rows=(), terms=(), background=(), history_days=0)


def explain(data, settings, lookup, themes=NO_THEMES):
    return build_explanation(data, settings, lookup=lookup, themes=themes)


class Lookup:
    """Stored daily_index rows by date; records which days were asked for."""

    def __init__(self, rows: dict[date, DayMood]):
        self.rows = rows
        self.asked: list[date] = []

    def __call__(self, day: date) -> DayMood | None:
        self.asked.append(day)
        return self.rows.get(day)


# --- previous session ---


def test_monday_skips_thin_sunday_and_uses_friday(settings):
    lookup = Lookup({SUN: stored(-40.0, 2), FRI: stored(-10.0, 15), THU: stored(5.0, 20)})
    prev = find_previous_session(MON, lookup, settings)
    assert prev == PreviousSession(FRI, -10.0, 15)
    assert lookup.asked == [SUN, SAT, FRI]  # newest first, stops at the first that qualifies


def test_weekend_day_qualifies_when_it_meets_the_floor(settings):
    lookup = Lookup({SUN: stored(-40.0, 5), FRI: stored(-10.0, 15)})
    assert find_previous_session(MON, lookup, settings).date == SUN


def test_nothing_in_lookback_qualifies(settings):
    day = date(2026, 10, 7)
    thin = {day - timedelta(days=k): stored(-30.0, 4) for k in range(1, 6)}
    lookup = Lookup({**thin, day - timedelta(days=6): stored(-30.0, 50)})  # D-6 is outside the lookback
    assert find_previous_session(day, lookup, settings) is None
    assert lookup.asked == [day - timedelta(days=k) for k in range(1, 6)]


def test_lookback_is_inclusive_of_d_minus_5(settings):
    day = date(2026, 10, 7)
    lookup = Lookup({day - timedelta(days=5): stored(12.0, 9)})
    assert find_previous_session(day, lookup, settings).date == date(2026, 10, 2)


def test_previous_mood_is_the_displayed_value(settings):
    lookup = Lookup({FRI: stored(-21.45, 15)})
    assert find_previous_session(SAT, lookup, settings).mood == -21.5  # half-up, as the header shows it


# --- change slot ---


@pytest.mark.parametrize(
    "prev_raw, shown",
    [
        # Today shows -26.4. Raw -21.44 displays -21.4: delta -5.0, at the floor, shown,
        # although the raw delta (-4.96) is under it.
        (-21.44, -5.0),
        # Raw -21.46 displays -21.5: delta -4.9, just under, omitted.
        (-21.46, None),
    ],
)
def test_change_uses_displayed_moods_at_and_under_the_floor(settings, prev_raw, shown):
    prev = find_previous_session(SAT, Lookup({FRI: stored(prev_raw, 15)}), settings)
    assert mood_change(-26.4, prev, settings) == shown


def test_change_omitted_without_previous_session(settings):
    assert mood_change(-26.4, None, settings) is None


# --- the sentence ---


def test_clustered_day_without_change_or_driver(settings):
    exp = explain(report(settings), settings, Lookup({}))
    assert exp.lede == "Financial news leaned moderately negative today (index -26.4) across 15 headlines, with no dominant theme."
    assert (exp.previous, exp.change, exp.driver) == (None, None, None)


def test_change_names_yesterday_or_the_session_date(settings):
    data = report(settings)  # Wed Sep 1, 2027
    lede = explain(data, settings, Lookup({date(2027, 8, 31): stored(-10.0, 12)})).lede
    assert "(index -26.4, down 16.4 from yesterday)" in lede
    lede = explain(data, settings, Lookup({date(2027, 8, 30): stored(-40.0, 12)})).lede
    assert "(index -26.4, up 13.6 from Mon Aug 30)" in lede


def test_flat_day_says_was_and_never_names_a_driver(settings):
    data = report(settings, mood=3.2)
    lede = build_lede(data, previous=None, change=None, driver=Driver(5, ("fed", "rates")), clustered=True)
    assert lede == "Financial news was roughly flat today (index +3.2) across 15 headlines, with no dominant theme."


def test_driver_named_when_given(settings):
    lede = build_lede(report(settings), previous=None, change=None, driver=Driver(4, ("tariff", "steel", "china")), clustered=True)
    assert lede.endswith("across 15 headlines, driven mainly by 4 headlines on tariff, steel and china.")


def test_one_headline_is_singular(settings):
    lede = build_lede(report(settings, mood=-91.0, n=1), previous=None, change=None, driver=None, clustered=True)
    assert "across 1 headline," in lede


def test_driver_stub_never_claims_a_driver_yet(settings):
    assert explain(report(settings), settings, Lookup({})).driver is None


def test_empty_day_has_no_explanation(settings):
    empty = report(settings)
    empty = ReportData(**{**empty.__dict__, "mood": None, "band": None, "n_headlines": 0})
    assert explain(empty, settings, Lookup({})) is None


def test_phrase_bank_varies_across_dates(settings):
    days = [date(2026, 10, 1) + timedelta(days=k) for k in range(30)]
    ledes = {build_lede(report(settings, day=d), previous=None, change=None, driver=None, clustered=True) for d in days}
    assert len({lede.split(" leaned")[0] for lede in ledes}) == 3  # every subject variant occurs


_VARIANTS_SCRIPT = """
from datetime import date, timedelta
from newsmood.config import get_settings
from newsmood.reporting.explain import build_lede, PreviousSession
from newsmood.reporting.report_data import ReportData, band_label
s = get_settings()
for k in range(20):
    d = date(2026, 10, 1) + timedelta(days=k)
    data = ReportData(d, -26.4, band_label(-26.4, s.lede_bands), 15, 2, 0, 0, 60, (), (), (), (), (), "m", "g")
    print(build_lede(data, previous=PreviousSession(d - timedelta(days=1), -10.0, 12), change=-16.4, driver=None, clustered=True))
"""


def test_same_date_same_variant_across_interpreters():
    # hash() of a str is salted per process (PYTHONHASHSEED). If the phrase
    # bank used it, two seeds would pick different variants for some of these
    # 20 dates.
    outputs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run([sys.executable, "-c", _VARIANTS_SCRIPT], env=env, capture_output=True, text=True, check=True)
        outputs.append(out.stdout)
    assert outputs[0] == outputs[1]
    assert len(set(outputs[0].splitlines())) > 1



# --- block 3 and the driver (real vectorizer, no store) ---

DAY = FIRST_VARIANTS_DAY
STEEL = [  # 5 headlines on one topic
    "Steel tariff imports slump as tariff fight widens",
    "Steel tariff imports fall on new tariff threat",
    "Tariff on steel imports hits mills",
    "Steel imports tariff squeeze deepens",
    "Steel tariff imports outlook dims",
]
BANKS = [
    "Bank deposits outflows accelerate at regional lenders",
    "Regional bank deposits outflows spook investors",
    "Deposits outflows hit regional bank shares",
]
CHIPS = [
    "Chip stocks rally on record AI orders",
    "AI chip orders lift chip stocks to record",
    "Record AI chip orders send chip stocks higher",
]
RETIREMENT = [
    "Retirement savings advice for couples near retirement",
    "Retirement savings: how couples should split accounts",
    "Couples retirement savings questions answered",
    "Retirement savings tips couples overlook",
]
SCATTERED = [
    "Oil slides on supply glut",
    "Airline cancels routes",
    "Chipmaker misses estimates",
    "Copper drops in Asia",
    "Retail sales disappoint",
    "Yen weakens sharply",
    "Shipping rates collapse",
    "Weather delays harvest",
    "Museum reopens downtown",
]


def hrow(id_, label, conf, title, published="2027-09-01T14:00:00.000000+00:00"):
    return {"id": id_, "label": label, "confidence": conf, "title": title, "url": None, "source": "cnbc", "published_at": published}


def day_rows(*groups):
    """groups: (label, conf, titles). Ids are unique and ordered by input."""
    out = []
    for label, conf, titles in groups:
        out += [hrow(f"h{len(out) + k:02d}", label, conf, t) for k, t in enumerate(titles)]
    return out


def themed(settings, rows, *, background=(), history_days=0, lookup=None):
    """The report data and explanation for `rows` on DAY, built as the pipeline does."""
    mood = compute_mood(rows, low_confidence=settings.index.low_confidence)
    data = build_report_data(DayResult(DAY, mood, MID, "recomputed"), rows, settings)
    themes = ThemeInputs(rows=rows, terms=mood.terms, background=tuple(background), history_days=history_days)
    return data, build_explanation(data, settings, lookup=lookup or Lookup({}), themes=themes)


def with_explain(settings, **changes):
    return settings.model_copy(update={"explain": settings.explain.model_copy(update=changes)})


def test_five_headlines_every_block_declines_and_lede_is_shortest(settings):
    # Done-when case: below explain.min_headlines nothing is clustered, so the
    # lede makes no theme claim at all, not even "no dominant theme".
    rows = day_rows(("negative", 0.9, BANKS), ("negative", 0.8, SCATTERED[:2]))
    data, exp = themed(settings, rows)
    assert not exp.clustered and exp.clusters == () and exp.driver is None
    assert exp.lede == "Financial news leaned strongly negative today (index -100.0) across 5 headlines."
    md = render(data, exp)
    assert "**Clusters:** no clear theme today (5 headlines; clustering needs 8)." in md
    assert "theme" not in exp.lede


def test_clustered_day_with_nothing_qualifying_says_no_dominant_theme(settings):
    rows = day_rows(("negative", 0.9, SCATTERED))  # 9 unrelated headlines
    data, exp = themed(settings, rows)
    assert exp.clustered and exp.clusters == ()
    assert exp.lede.endswith("across 9 headlines, with no dominant theme.")
    assert "**Clusters:** no clear theme today.\n" in render(data, exp)


def test_oct2_style_neutral_cluster_is_shown_but_never_the_driver(settings):
    # The largest cluster is 4 neutral retirement columns: share 0, so no driver.
    rows = day_rows(("neutral", 0.97, RETIREMENT), ("negative", 0.9, SCATTERED[:7]))
    data, exp = themed(settings, rows)
    assert [(c.size, c.n_neutral, c.share_pct) for c in exp.clusters] == [(4, 4, 0.0)]
    assert "retirement" in exp.clusters[0].name
    assert exp.driver is None
    assert exp.lede.endswith("across 11 headlines, with no dominant theme.")


def test_driver_is_largest_share_in_mood_direction_not_largest_cluster(settings):
    # Steel: 5 headlines, 4 negative at 0.40 + 1 neutral (purity 0.8, tilt -1.6).
    # Banks: 3 negatives at 0.95 (tilt -2.85). Largest by size is steel; the
    # driver must be banks.
    rows = day_rows(
        ("negative", 0.40, STEEL[:4]), ("neutral", 0.40, STEEL[4:]), ("negative", 0.95, BANKS),
        ("positive", 0.60, SCATTERED[:2]),
    )
    data, exp = themed(settings, rows)
    sizes = {c.ids[0]: c.size for c in exp.clusters}
    assert sorted(sizes.values()) == [3, 5]
    assert exp.driver is not None and exp.driver.n_headlines == 3
    assert "deposits" in exp.driver.terms and "steel" not in exp.driver.terms
    assert "driven mainly by 3 headlines on " in exp.lede
    # Movement Σ|conf·s| = 1.6 + 2.85 + 1.2 = 5.65.
    assert [c.share_pct for c in exp.clusters] == [-50.4, -28.3]


def test_flat_day_shares_stay_within_100_percent(settings):
    # Σ(conf·s) = 0: a naive share denominator would divide by zero. Over
    # Σ|conf·s| = 5.4 each side is exactly half.
    rows = day_rows(("negative", 0.9, BANKS), ("positive", 0.9, CHIPS), ("neutral", 0.7, SCATTERED[:2]))
    data, exp = themed(settings, rows)
    assert data.mood == 0.0 and data.band == "roughly flat"
    assert sorted(c.share_pct for c in exp.clusters) == [-50.0, 50.0]
    assert all(abs(c.share_pct) <= 100 for c in exp.clusters)
    assert exp.driver is None
    assert exp.lede.startswith("Financial news was roughly flat today (index +0.0)")


def test_window_fit_includes_today(settings):
    # "truce" appears in 3 of today's headlines and nowhere in the background.
    # Fit on the background alone, it would be out of vocabulary, the rows
    # would be all zeros, and no cluster could form.
    today = ["Truce talks lift ceasefire hopes", "Ceasefire truce talks stall", "Truce talks resume as ceasefire holds"]
    background = [hrow(f"b{i}", "neutral", 0.9, t, "2027-08-20T14:00:00.000000+00:00") for i, t in enumerate(SCATTERED + STEEL)]
    rows = day_rows(("negative", 0.9, today), ("neutral", 0.9, SCATTERED[:6]))
    _, exp = themed(settings, rows, background=background)
    assert len(exp.clusters) == 1 and "truce" in exp.clusters[0].name
    assert exp.fit.n_headlines == len(rows) + len(background)


def test_cluster_name_uses_its_own_rows(settings):
    # Day-level top terms would give both clusters the same name.
    rows = day_rows(("negative", 0.9, BANKS), ("negative", 0.8, STEEL[:3]), ("neutral", 0.7, SCATTERED[:3]))
    _, exp = themed(settings, rows)
    names = {c.ids[0]: set(c.name) for c in exp.clusters}
    banks, steel = names["h00"], names["h03"]
    assert banks and steel and not banks & steel
    assert "steel" not in banks and "deposits" not in steel


def test_cluster_name_needs_a_term_in_at_least_two_rows(settings):
    # Full history, so names rank by lift. "zinc" and "deepens" are each in one
    # row; the shared words also fill part of the background, so their lift
    # is lower than a one-row word's. One headline is not a theme.
    titles = ["Steel tariff imports slump zinc", "Steel tariff imports slump", "Steel tariff imports slump deepens"]
    background = [
        hrow(f"b{i}", "neutral", 0.9, t, "2027-08-20T14:00:00.000000+00:00")
        for i, t in enumerate([*STEEL, *[f"Steel tariff imports slump {w}" for w in ("again", "widens", "eases")],
                               "Zinc mine opens", "Squeeze deepens at mills", *SCATTERED])
    ]
    rows = day_rows(("negative", 0.9, titles), ("neutral", 0.9, SCATTERED[:6]))
    single_row = {"zinc", "deepens", "slump zinc", "slump deepens"}
    _, exp = themed(settings, rows, background=background, history_days=30)
    (cluster,) = exp.clusters
    assert cluster.name and not set(cluster.name) & single_row
    # Control: with the floor at 1 a one-row word outranks them, so the floor is what keeps it out.
    _, loose = themed(with_explain(settings, min_term_headlines=1), rows, background=background, history_days=30)
    assert set(loose.clusters[0].name) & single_row


def test_input_order_does_not_change_the_explanation(settings):
    rows = day_rows(("negative", 0.40, STEEL[:4]), ("neutral", 0.40, STEEL[4:]), ("negative", 0.95, BANKS), ("positive", 0.6, SCATTERED[:2]))
    _, forward = themed(settings, rows)
    _, backward = themed(settings, list(reversed(rows)))
    assert forward == backward


# --- vectorizer modes ---


def test_window_mode_reports_fit_line_and_no_oov_line(settings):
    background = [hrow("b0", "neutral", 0.9, "Copper drops in Asia", "2027-08-20T14:00:00.000000+00:00")]
    rows = day_rows(("negative", 0.9, SCATTERED))
    data, exp = themed(settings, rows, background=background)
    assert exp.oov is None
    md = render(data, exp)
    assert "_Themes use TF-IDF fit on 10 headlines (2027-08-20–2027-09-01)._" in md
    assert "outside it" not in md


@pytest.fixture
def phrasebank_settings(settings, tmp_path):
    vec = TfidfVectorizer().fit(["bank deposits outflows", "regional bank shares", "deposits fall", "steel tariff"])
    joblib.dump(vec, tmp_path / "tfidf_vectorizer.joblib")
    s = with_explain(settings, vectorizer="phrasebank")
    return s.model_copy(update={"logreg": s.logreg.model_copy(update={"artifact_dir": str(tmp_path)})})


def test_phrasebank_mode_reports_oov_and_flags_advice(phrasebank_settings):
    rows = day_rows(("negative", 0.9, BANKS), ("neutral", 0.9, SCATTERED[:5]))
    data, exp = themed(phrasebank_settings, rows)
    assert exp.fit is None and exp.oov.vectorizer == "phrasebank"
    assert exp.oov.rate_pct > 20 and exp.oov.suggest_window
    assert [c.size for c in exp.clusters] == [3]  # clusters still run on the PhraseBank vocabulary
    md = render(data, exp)
    assert "PhraseBank baseline's vocabulary" in md and "of this day's words" in md
    assert "explain.vectorizer" not in md  # advice is stderr's job


def test_phrasebank_mode_without_the_file_names_window(settings, tmp_path):
    s = with_explain(settings, vectorizer="phrasebank")
    s = s.model_copy(update={"logreg": s.logreg.model_copy(update={"artifact_dir": str(tmp_path / "missing")})})
    with pytest.raises(ExplainError, match="explain.vectorizer: window"):
        themed(s, day_rows(("negative", 0.9, SCATTERED)))
