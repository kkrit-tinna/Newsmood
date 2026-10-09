"""Report data builder (T4.2): plain objects, no DB, no network."""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from newsmood.config import LedeBandsSettings, get_settings
from newsmood.reporting.aggregate import DayResult, compute_mood
from newsmood.reporting.report_data import band_label, build_report_data, is_linkable

DAY = date(2026, 10, 1)
MID = "owner/model@" + "a" * 40
BANDS = LedeBandsSettings(flat=5, mild=15, moderate=35)


@pytest.fixture
def settings():
    return get_settings()


def row(id_, label, conf, source="cnbc", published="2026-10-01T14:00:00.000000+00:00", url=None, title=None):
    return {
        "id": id_,
        "label": label,
        "confidence": conf,
        "source": source,
        "published_at": published,
        "url": f"https://example.com/{id_}" if url is None else url,
        "title": title or f"headline {id_}",
    }


def build(rows, settings):
    result = DayResult(DAY, compute_mood(rows, low_confidence=settings.index.low_confidence), MID, "recomputed")
    return build_report_data(result, rows, settings)


def mixed_day():
    return [
        row("n1", "negative", 0.91, "cnbc"),
        row("n2", "negative", 0.55, "marketwatch"),
        row("u1", "neutral", 0.88, "cnbc"),
        row("u2", "neutral", 0.70, "yahoo"),
        row("u3", "neutral", 0.66, "cnbc"),
        row("p1", "positive", 0.97, "yahoo"),
        row("p2", "positive", 0.81, "cnbc"),
        row("p3", "positive", 0.59, "marketwatch"),
    ]


# --- counts and shares ---


def test_counts_sum_to_n_and_shares_match_counts(settings):
    data = build(mixed_day(), settings)
    assert data.n_headlines == 8
    assert [(s.sentiment, s.count) for s in data.slices] == [("negative", 2), ("positive", 3), ("neutral", 3)]
    assert sum(s.count for s in data.slices) == data.n_headlines
    assert [s.share_pct for s in data.slices] == [25.0, 37.5, 37.5]
    assert data.n_sources == 3
    assert data.n_unsure == 2  # 0.55 and 0.59 < 0.60
    assert data.low_confidence_pct == 60


def test_slice_colors_follow_sentiment_from_config(settings):
    colors = settings.report.colors
    data = build(mixed_day(), settings)
    assert {s.sentiment: s.color for s in data.slices} == {
        "negative": colors.negative, "positive": colors.positive, "neutral": colors.neutral
    }


def test_pie_drops_zero_slices_and_orders_by_count_then_sentiment(settings):
    # no negatives: neutral 3, positive 1 → neutral first, each with its own color
    rows = [row("p1", "positive", 0.9)] + [row(f"u{i}", "neutral", 0.8) for i in range(3)]
    data = build(rows, settings)
    assert [(s.sentiment, s.color) for s in data.pie] == [
        ("neutral", settings.report.colors.neutral),
        ("positive", settings.report.colors.positive),
    ]
    assert len(data.slices) == 3  # the text line still names all three
    # ties keep SENTIMENT_ORDER: negative, positive, neutral
    tied = build(mixed_day()[:2] + mixed_day()[2:4] + mixed_day()[5:7], settings)
    assert [s.sentiment for s in tied.pie] == ["negative", "positive", "neutral"]


def test_shares_round_half_up_to_one_decimal(settings):
    # 1/16 = 6.25% → 6.3; 15/16 = 93.75% → 93.8 (half-up, not half-even)
    rows = [row("n0", "negative", 0.9)] + [row(f"u{i}", "neutral", 0.9) for i in range(15)]
    data = build(rows, settings)
    shares = {s.sentiment: s.share_pct for s in data.slices}
    assert shares == {"negative": 6.3, "positive": 0.0, "neutral": 93.8}


def test_mood_taken_from_aggregate_not_recomputed(settings):
    rows = mixed_day()
    mood = compute_mood(rows, low_confidence=settings.index.low_confidence)
    fake = dataclasses.replace(mood, mood=-12.345)
    data = build_report_data(DayResult(DAY, fake, MID, "kept"), rows, settings)
    assert data.mood == -12.3
    assert data.band == "mildly negative"
    assert data.model_id == MID
    assert data.gates is None  # not evaluated outside the pipeline


def test_neutral_headlines_are_in_terms_with_term_zero():
    mood = compute_mood(mixed_day(), low_confidence=0.6)
    terms = dict(mood.terms)
    assert set(terms) == {r["id"] for r in mixed_day()}
    assert [terms[i] for i in ("u1", "u2", "u3")] == [0.0, 0.0, 0.0]


def test_late_headlines_for_frozen_day_are_excluded_and_counted(settings):
    frozen = mixed_day()
    result = DayResult(DAY, compute_mood(frozen, low_confidence=0.6), MID, "kept")
    late = [row("late1", "negative", 0.99), row("late2", "positive", 0.98, source="newwire")]
    data = build_report_data(result, frozen + late, settings)
    assert data.n_late == 2
    assert data == dataclasses.replace(build(frozen, settings), n_late=2)
    assert "late1" not in {h.id for h in data.headlines}
    assert "newwire" not in {s.source for s in data.sources}
    assert data.mood == round(result.day.mood, 1)


def test_no_late_headlines_counts_zero(settings):
    assert build(mixed_day(), settings).n_late == 0


def test_stored_id_without_row_raises(settings):
    rows = mixed_day()
    result = DayResult(DAY, compute_mood(rows, low_confidence=0.6), MID, "kept")
    with pytest.raises(ValueError, match="no row"):
        build_report_data(result, rows[:-1], settings)


def test_row_labels_not_matching_snapshot_raise(settings):
    rows = mixed_day()
    result = DayResult(DAY, compute_mood(rows, low_confidence=0.6), MID, "kept")
    rows[0] = {**rows[0], "label": "positive"}
    with pytest.raises(ValueError, match="labels"):
        build_report_data(result, rows, settings)


def test_negative_zero_mood_prints_as_zero(settings):
    rows = mixed_day()
    fake = dataclasses.replace(compute_mood(rows, low_confidence=0.6), mood=-0.04)
    data = build_report_data(DayResult(DAY, fake, MID, "kept"), rows, settings)
    assert data.mood == 0.0 and str(data.mood) == "0.0"
    assert data.band == "roughly flat"


def test_result_is_immutable(settings):
    data = build(mixed_day(), settings)
    with pytest.raises(dataclasses.FrozenInstanceError):
        data.mood = 1.0
    assert isinstance(data.headlines, tuple)


# --- band ---


@pytest.mark.parametrize(
    "mood, band",
    [
        (0.0, "roughly flat"),
        (4.9, "roughly flat"),
        (5.0, "mildly positive"),
        (14.9, "mildly positive"),
        (15.0, "moderately positive"),
        (34.9, "moderately positive"),
        (35.0, "strongly positive"),
        (100.0, "strongly positive"),
        (-4.9, "roughly flat"),
        (-5.0, "mildly negative"),
        (-14.9, "mildly negative"),
        (-15.0, "moderately negative"),
        (-34.9, "moderately negative"),
        (-35.0, "strongly negative"),
        (None, None),
    ],
)
def test_band_boundaries(mood, band):
    assert band_label(mood, BANDS) == band


def test_band_uses_displayed_mood(settings):
    # 4.96 displays as +5.0, so it must not be labelled "roughly flat"
    rows = mixed_day()
    fake = dataclasses.replace(compute_mood(rows, low_confidence=0.6), mood=4.96)
    data = build_report_data(DayResult(DAY, fake, MID, "kept"), rows, settings)
    assert (data.mood, data.band) == (5.0, "mildly positive")


# --- recommended articles ---


def test_recommended_top_k_per_sentiment(settings):
    rows = mixed_day() + [row("n3", "negative", 0.99), row("n4", "negative", 0.30)]
    data = build(rows, settings)
    groups = {g.sentiment: g for g in data.recommended}
    assert [g.sentiment for g in data.recommended] == ["negative", "positive", "neutral"]
    assert [h.id for h in groups["negative"].items] == ["n3", "n1", "n2"]
    assert [h.id for h in groups["positive"].items] == ["p1", "p2", "p3"]
    assert [h.id for h in groups["neutral"].items] == ["u1", "u2"]
    assert not any(g.shortfall for g in data.recommended)
    top = groups["negative"].items[0]
    assert (top.certainty_pct, top.source, top.title, top.url) == (99, "cnbc", "headline n3", "https://example.com/n3")


def test_recommended_sizes_come_from_config(settings):
    s = settings.model_copy(
        update={"report": settings.report.model_copy(update={"recommended": settings.report.recommended.model_copy(update={"neutral": 1})})}
    )
    data = build(mixed_day(), s)
    assert [len(g.items) for g in data.recommended] == [2, 3, 1]


def test_recommended_shortfall_flagged(settings):
    rows = [row("n1", "negative", 0.9), row("n2", "negative", 0.8), row("p1", "positive", 0.9)] + [
        row(f"u{i}", "neutral", 0.7) for i in range(4)
    ]
    groups = {g.sentiment: g for g in build(rows, settings).recommended}
    assert (groups["negative"].requested, len(groups["negative"].items), groups["negative"].shortfall) == (3, 2, True)
    assert (len(groups["positive"].items), groups["positive"].shortfall) == (1, True)
    assert groups["neutral"].shortfall is False


def test_recommended_zero_of_a_sentiment_is_present_and_flagged(settings):
    rows = [row("u1", "neutral", 0.9), row("p1", "positive", 0.9)]
    groups = {g.sentiment: g for g in build(rows, settings).recommended}
    assert groups["negative"].items == () and groups["negative"].shortfall


def test_confidence_ties_broken_by_published_at_then_id(settings):
    rows = [
        row("b", "negative", 0.9, published="2026-10-01T15:00:00.000000+00:00"),
        row("c", "negative", 0.9, published="2026-10-01T13:00:00.000000+00:00"),
        row("a", "negative", 0.9, published="2026-10-01T15:00:00.000000+00:00"),
        row("d", "negative", 0.9, published=None),
    ]
    data = build(rows, settings)
    assert [h.id for h in data.recommended[0].items] == ["c", "a", "b"]
    assert [h.id for h in data.headlines] == ["c", "a", "b", "d"]


def test_order_does_not_depend_on_input_order(settings):
    rows = mixed_day()
    assert build(rows, settings) == build(list(reversed(rows)), settings)


# --- full list ---


def test_full_list_negative_positive_neutral_then_confidence_desc(settings):
    data = build(mixed_day(), settings)
    assert [h.id for h in data.headlines] == ["n1", "n2", "p1", "p2", "p3", "u1", "u2", "u3"]
    assert [h.sentiment for h in data.headlines] == ["negative"] * 2 + ["positive"] * 3 + ["neutral"] * 3
    assert [h.unsure for h in data.headlines] == [False, True, False, False, True, False, False, False]


# --- sources ---


def test_sources_table_order_and_counts(settings):
    data = build(mixed_day(), settings)
    assert [(s.source, s.total, s.negative, s.neutral, s.positive) for s in data.sources] == [
        ("cnbc", 4, 1, 2, 1),
        ("marketwatch", 2, 1, 0, 1),
        ("yahoo", 2, 0, 1, 1),
    ]
    assert sum(s.total for s in data.sources) == data.n_headlines


def test_missing_source_shows_unknown(settings):
    rows = [row("a", "neutral", 0.9, source=None), row("b", "neutral", 0.8, source="  "), row("c", "positive", 0.7)]
    data = build(rows, settings)
    assert [h.source for h in data.headlines] == ["cnbc", "unknown", "unknown"]
    assert [(s.source, s.total) for s in data.sources] == [("unknown", 2), ("cnbc", 1)]
    assert data.n_sources == 2


# --- urls ---


@pytest.mark.parametrize(
    "url, linked",
    [
        ("https://www.cnbc.com/2026/10/01/a.html?utm=x#frag", True),
        ("http://example.com/a", True),
        ("HTTPS://EXAMPLE.COM/a", True),
        ("", False),
        ("javascript:alert(1)", False),
        ("ftp://example.com/file", False),
        ("mailto:a@example.com", False),
        ("/relative/path", False),
        ("https:no-host", False),
    ],
)
def test_url_linkability(url, linked):
    assert is_linkable(url) is linked


def test_urls_passed_through_unchanged(settings):
    weird = "https://example.com/a b?x=1&y=<2>#(c)"
    rows = [
        row("a", "neutral", 0.9, url=weird),
        row("b", "neutral", 0.8, url="javascript:alert(1)"),
        {k: v for k, v in row("c", "neutral", 0.7).items() if k != "url"},
    ]
    data = build(rows, settings)
    by_id = {h.id: h for h in data.headlines}
    assert (by_id["a"].url, by_id["a"].linked) == (weird, True)
    assert (by_id["b"].url, by_id["b"].linked) == ("javascript:alert(1)", False)
    assert (by_id["c"].url, by_id["c"].linked) == (None, False)


# --- certainty ---


@pytest.mark.parametrize(
    "conf, pct",
    [(0.955, 96), (0.945, 95), (0.9549, 95), (0.999, 100), (0.5, 50), (0.333, 33), (1.0, 100), (0.334, 33)],
)
def test_certainty_percent_rounds_half_up(settings, conf, pct):
    data = build([row("a", "neutral", conf)], settings)
    assert data.headlines[0].certainty_pct == pct


# --- empty day ---


def test_empty_day(settings):
    data = build([], settings)
    assert data.mood is None and data.band is None
    assert data.is_empty
    assert (data.n_headlines, data.n_sources, data.n_unsure, data.n_late) == (0, 0, 0, 0)
    assert data.slices == data.pie == data.recommended == data.sources == data.headlines == ()
    assert data.date == DAY and data.model_id == MID


# --- contribution (T4.2b block 1) ---


def test_contribution_is_conf_over_total_movement(settings):
    # Σ|conf·s| over non-neutrals = 0.91 + 0.55 + 0.97 + 0.81 + 0.59 = 3.83
    data = build(mixed_day(), settings)
    got = {h.id: h.contribution_pct for h in data.headlines}
    assert got == {"n1": 23.8, "n2": 14.4, "p1": 25.3, "p2": 21.1, "p3": 15.4, "u1": None, "u2": None, "u3": None}
    shown = [h.id for g in data.recommended for h in g.items if h.contribution_pct is not None]
    assert shown == ["n1", "n2", "p1", "p2", "p3"]


def test_contribution_rounds_half_up(settings):
    # 100 × 0.49 / 4.0 = 12.25: half-up gives 12.3, round() would give 12.2.
    rows = [row("n1", "negative", 0.49)] + [row(f"p{i}", "positive", c) for i, c in enumerate((0.9, 0.9, 0.9, 0.81))]
    assert build(rows, settings).headlines[0].contribution_pct == 12.3


def test_contribution_is_not_normalized(settings):
    # Three equal negatives: 33.3 each, summing to 99.9. Normalizing would bump one to 33.4.
    rows = [row(f"n{i}", "negative", 0.8) for i in range(3)]
    assert [h.contribution_pct for h in build(rows, settings).headlines] == [33.3, 33.3, 33.3]


def test_contribution_comes_from_the_stored_terms(settings):
    # A frozen day's snapshot holds its own terms; the shares follow them,
    # not the rows' current confidences.
    rows = mixed_day()
    mood = compute_mood(rows, low_confidence=0.6)
    snapshot = dataclasses.replace(mood, terms=tuple((i, -1.0 if i == "n1" else 0.0) for i, _ in mood.terms))
    data = build_report_data(DayResult(DAY, snapshot, MID, "kept"), rows, settings)
    assert {h.id: h.contribution_pct for h in data.headlines if h.contribution_pct is not None} == {"n1": 100.0}


def test_all_neutral_day_has_no_contributions(settings):
    data = build([row(f"u{i}", "neutral", 0.8) for i in range(3)], settings)
    assert [h.contribution_pct for h in data.headlines] == [None, None, None]
