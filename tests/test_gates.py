"""Quality gates (T3.5). Each gate gets failing data and must catch it; a
gate that is removed or inverted fails a test here.

Two layers: evaluate() on hand-built inputs (every status, every boundary),
then run_report() end to end with fake feeds, a fake classifier and a fixed
clock, for what a block does to files, the index and the CLI.
"""

from __future__ import annotations

import dataclasses
import json
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from newsmood import cli
from newsmood.config import FeedSettings, StoreSettings, get_settings
from newsmood.data import feeds, store
from newsmood.evaluation import gates
from newsmood.evaluation.gates import BLOCK, DISCLOSE, FAIL, FIRED, NOT_APPLICABLE, PASS, DayStats, FeedStatus, GateInputs
from newsmood.models import scoring
from newsmood.reporting import pipeline
from newsmood.reporting.template import render
from tests.test_pipeline import NOW, URL_A, URL_B, FakeClassifier, FakeFeeds, utc
from tests.test_template import report as template_report

GATE_ORDER = ["pipeline_fresh", "all_scored", "thin_day", "feeds_down", "single_source", "low_confidence", "stale_feed"]
H = timedelta(hours=1)


# --- evaluate() on hand-built inputs ---


def inputs(**overrides) -> GateInputs:
    """A healthy run: everything passes unless a test overrides it."""
    base = GateInputs(
        is_today=True,
        now=NOW,
        last_fetched_at=NOW - H,
        n_unscored=0,
        feeds=(FeedStatus("cnbc-finance", True, NOW - 2 * H), FeedStatus("marketwatch-top", True, NOW - 3 * H)),
        day=DayStats(n_headlines=10, source_counts={"cnbc-finance": 5, "marketwatch-top": 5}, n_low_conf=2),
    )
    return dataclasses.replace(base, **overrides)


def evaluate(**overrides) -> gates.GateReport:
    return gates.evaluate(inputs(**overrides), get_settings())


def gate(report: gates.GateReport, name: str) -> gates.GateResult:
    (r,) = [r for r in report.results if r.name == name]
    return r


def test_healthy_run_passes_every_gate_in_fixed_order():
    report = evaluate()
    assert [r.name for r in report.results] == GATE_ORDER
    assert [r.kind for r in report.results] == [BLOCK, BLOCK] + [DISCLOSE] * 5
    assert {r.status for r in report.results} == {PASS}
    assert not report.blocked
    assert report.summary() == gates.GateSummary(n_passed=2, n_applicable=2, n_not_applicable=0, notes=())


def test_thresholds_come_from_config():
    s = get_settings()
    report = gates.evaluate(inputs(), s)
    assert gate(report, "pipeline_fresh").threshold == s.gates.max_feed_staleness_hours == 36
    assert gate(report, "stale_feed").threshold == s.gates.max_feed_staleness_hours
    assert gate(report, "single_source").threshold == s.gates.max_single_source_share == 0.70
    assert gate(report, "low_confidence").threshold == s.gates.max_low_confidence_rate == 0.50
    # thin_day reads the explain floor itself: one key, not a copy of it.
    assert gate(report, "thin_day").threshold == s.explain.min_headlines == 8
    assert gate(report, "thin_day").threshold_key == "explain.min_headlines"


# 1. pipeline_fresh (block)


def test_pipeline_fresh_fails_when_newest_fetch_is_older_than_max():
    report = evaluate(last_fetched_at=NOW - 37 * H)
    r = gate(report, "pipeline_fresh")
    assert r.status == FAIL
    assert r.observed == {"newest_fetched_at": "2026-10-01T05:00:00.000000+00:00"}
    assert report.blocked and report.failed == (r,)


def test_pipeline_fresh_fails_on_an_empty_store():
    assert gate(evaluate(last_fetched_at=None), "pipeline_fresh").status == FAIL


@pytest.mark.parametrize("age, status", [(36 * H - timedelta(seconds=1), PASS), (36 * H, FAIL)])
def test_pipeline_fresh_boundary_is_age_at_least_max(age, status):
    assert gate(evaluate(last_fetched_at=NOW - age), "pipeline_fresh").status == status


def test_pipeline_fresh_zero_hours_fails_even_on_a_row_fetched_this_run():
    # T4.4's forced-failure override relies on this: age 0 >= 0.
    s = get_settings()
    s = s.model_copy(update={"gates": s.gates.model_copy(update={"max_feed_staleness_hours": 0})})
    report = gates.evaluate(inputs(last_fetched_at=NOW), s)
    assert gate(report, "pipeline_fresh").status == FAIL


def test_pipeline_fresh_not_applicable_on_a_past_date():
    report = evaluate(is_today=False, last_fetched_at=NOW - 30 * 24 * H)
    r = gate(report, "pipeline_fresh")
    assert r.status == NOT_APPLICABLE and r.observed is None
    assert not report.blocked
    assert report.summary().n_applicable == 1 and report.summary().n_not_applicable == 1


# 2. all_scored (block)


def test_all_scored_fails_on_one_unscored_row():
    report = evaluate(n_unscored=1)
    r = gate(report, "all_scored")
    assert r.status == FAIL and r.observed == {"unscored": 1}
    assert report.blocked


def test_all_scored_still_judged_on_a_past_date():
    assert gate(evaluate(is_today=False, n_unscored=2), "all_scored").status == FAIL


# 3. thin_day (disclose)


@pytest.mark.parametrize("n, status", [(7, FIRED), (8, PASS)])
def test_thin_day_fires_below_explain_min_headlines(n, status):
    day = DayStats(n, {"cnbc-finance": n // 2, "marketwatch-top": n - n // 2}, 0)
    report = evaluate(day=day)
    assert gate(report, "thin_day").status == status
    assert not report.blocked


def test_thin_day_not_applicable_on_an_empty_day():
    # The empty report already says "No headlines for this date."
    assert gate(evaluate(day=DayStats(0, {}, 0)), "thin_day").status == NOT_APPLICABLE


# 4. feeds_down (disclose)


def test_feeds_down_fires_when_a_configured_feed_failed():
    feeds = (FeedStatus("cnbc-finance", True, NOW - H), FeedStatus("marketwatch-top", False, None))
    report = evaluate(feeds=feeds)
    r = gate(report, "feeds_down")
    assert r.status == FIRED
    assert r.observed == {"configured": 2, "failed": ["marketwatch-top"], "responded": 1}
    assert not report.blocked


def test_feeds_down_fires_when_every_feed_failed_but_does_not_block():
    feeds = (FeedStatus("cnbc-finance", False, None), FeedStatus("marketwatch-top", False, None))
    report = evaluate(feeds=feeds)
    assert gate(report, "feeds_down").observed["responded"] == 0
    assert not report.blocked  # pipeline_fresh decides that, from the store


@pytest.mark.parametrize("overrides", [{"feeds": None}, {"is_today": False}])
def test_feeds_down_not_applicable_without_ingest_or_on_a_past_date(overrides):
    feeds = (FeedStatus("cnbc-finance", False, None),)
    assert gate(evaluate(**{"feeds": feeds, **overrides}), "feeds_down").status == NOT_APPLICABLE


# 5. single_source (disclose)


def test_single_source_fires_on_the_oct_2_shape():
    # Oct 2: MarketWatch 11 of 15.
    day = DayStats(15, {"cnbc-finance": 4, "marketwatch-top": 11}, 2)
    report = evaluate(day=day)
    r = gate(report, "single_source")
    assert r.status == FIRED
    assert r.observed == {"count": 11, "headlines": 15, "share_pct": 73, "source": "marketwatch-top"}
    assert not report.blocked


def test_single_source_exactly_at_max_does_not_fire():
    day = DayStats(10, {"cnbc-finance": 3, "marketwatch-top": 7}, 0)
    assert gate(evaluate(day=day), "single_source").status == PASS


def test_single_source_tie_names_the_alphabetically_first_source():
    day = DayStats(2, {"z-feed": 1, "a-feed": 1}, 0)
    assert gate(evaluate(day=day), "single_source").observed["source"] == "a-feed"


# 6. low_confidence (disclose)


def test_low_confidence_fires_above_max_rate():
    day = DayStats(13, {"cnbc-finance": 7, "marketwatch-top": 6}, 8)
    r = gate(evaluate(day=day), "low_confidence")
    assert r.status == FIRED and r.observed == {"headlines": 13, "unsure": 8}


def test_low_confidence_exactly_at_max_does_not_fire():
    day = DayStats(10, {"cnbc-finance": 5, "marketwatch-top": 5}, 5)
    assert gate(evaluate(day=day), "low_confidence").status == PASS


# 7. stale_feed (disclose)


def test_stale_feed_fires_for_a_responding_feed_with_nothing_new():
    feeds = (
        FeedStatus("cnbc-finance", True, NOW - 40 * H),
        FeedStatus("marketwatch-top", True, NOW - H),
        FeedStatus("dead-feed", False, None),  # failed feeds are feeds_down's business
    )
    report = evaluate(feeds=feeds)
    r = gate(report, "stale_feed")
    assert r.status == FIRED
    assert r.observed == {
        "newest_published_at": {
            "cnbc-finance": "2026-10-01T02:00:00.000000+00:00",
            "marketwatch-top": "2026-10-02T17:00:00.000000+00:00",
        },
        "stale": ["cnbc-finance"],
    }
    assert not report.blocked


def test_stale_feed_not_applicable_on_a_past_date():
    feeds = (FeedStatus("cnbc-finance", True, NOW - 40 * H),)
    assert gate(evaluate(feeds=feeds, is_today=False), "stale_feed").status == NOT_APPLICABLE


# --- together ---


def test_every_disclosure_firing_still_does_not_block():
    report = evaluate(
        feeds=(FeedStatus("cnbc-finance", True, NOW - 40 * H), FeedStatus("marketwatch-top", False, None)),
        day=DayStats(5, {"cnbc-finance": 5}, 4),
    )
    assert [r.name for r in report.summary().notes] == ["thin_day", "feeds_down", "single_source", "low_confidence", "stale_feed"]
    assert not report.blocked
    assert report.summary().n_passed == 2


def test_day_gates_not_applicable_until_the_day_is_given():
    report = evaluate(day=None)
    assert {gate(report, n).status for n in ("thin_day", "single_source", "low_confidence")} == {NOT_APPLICABLE}


def test_quality_json_sorted_and_deterministic_apart_from_run_at():
    report = evaluate(n_unscored=3)
    a = json.loads(report.to_json("2026-10-02", NOW))
    b = json.loads(report.to_json("2026-10-02", NOW + H))
    assert a["run_at"] != b["run_at"]
    assert {**a, "run_at": None} == {**b, "run_at": None}
    assert report.to_json("2026-10-02", NOW) == json.dumps(a, indent=2, sort_keys=True) + "\n"
    assert a["blocked"] is True
    assert [g["name"] for g in a["gates"]] == GATE_ORDER
    assert set(a["gates"][0]) == {"kind", "name", "observed", "status", "threshold", "threshold_key"}


# --- the report's data-notes line and footer ---


def test_every_fired_disclosure_renders_as_one_phrase_on_one_line():
    report = evaluate(
        feeds=(FeedStatus("cnbc-finance", True, NOW - 40 * H), FeedStatus("marketwatch-top", False, None)),
        day=DayStats(6, {"cnbc-finance": 5, "marketwatch-top": 1}, 4),
    )
    text = render(template_report(gates=report.summary()))
    notes = (
        "_Data notes: Thin day: 6 headlines. 1 of 2 feeds responded. 83% of headlines from cnbc-finance. "
        "The model was unsure on 4 of 6. cnbc-finance had nothing newer than 36 h._"
    )
    assert notes in text
    assert "Gates: 2/2 passed · 5 data notes · Not financial advice." in text


def test_empty_day_still_shows_feed_notes_under_no_headlines():
    report = evaluate(feeds=(FeedStatus("cnbc-finance", False, None), FeedStatus("marketwatch-top", True, NOW - H)),
                      day=DayStats(0, {}, 0))
    empty = template_report(mood=None, band=None, n_headlines=0, gates=report.summary())
    text = render(empty)
    assert "No headlines for this date.\n\n_Data notes: 1 of 2 feeds responded._" in text
    assert "Thin day" not in text  # the empty line already says it


def test_past_date_footer_counts_pipeline_fresh_as_not_applicable():
    text = render(template_report(gates=evaluate(is_today=False).summary()))
    assert "Gates: 1/1 passed, 1 not applicable · Not financial advice." in text
    assert "Data notes" not in text


# --- end to end through run_report ---


@pytest.fixture
def settings(tmp_path):
    base = get_settings()
    ingest = base.ingest.model_copy(
        update={"delay_seconds": 0, "feeds": [FeedSettings(name="feed-a", url=URL_A), FeedSettings(name="feed-b", url=URL_B)]}
    )
    return base.model_copy(update={"store": StoreSettings(db_path=str(tmp_path / "test.db")), "ingest": ingest})


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path / "reports"


@pytest.fixture
def classifier():
    return FakeClassifier()


def balanced_feeds(per_feed: int = 4) -> FakeFeeds:
    """A healthy Oct 2 (ET): per_feed confident headlines from each feed."""
    f = FakeFeeds()
    for url in (URL_A, URL_B):
        for i in range(per_feed):
            word = "Up" if i % 2 else "Down"
            f.add(url, f"{word} {url[8]} story {i}", utc(f"2026-10-02T{9 + i:02d}:00:00-04:00"))
    return f


def run(settings, fake_feeds, classifier, out_dir, *, now=NOW, date_arg=None):
    return pipeline.run_report(
        settings, date_arg=date_arg, fetch=fake_feeds, get_classifier=lambda: classifier, now=lambda: now, out_dir=out_dir
    )


def quality(out_dir, day="2026-10-02") -> dict:
    return json.loads((out_dir / "quality" / f"{day}.json").read_text(encoding="utf-8"))


def test_all_pass_run_has_gates_footer_and_no_data_notes(settings, classifier, out_dir):
    r = run(settings, balanced_feeds(), classifier, out_dir)
    text = r.path.read_text(encoding="utf-8")
    assert "Gates: 2/2 passed · Not financial advice." in text
    assert "Data notes" not in text
    q = quality(out_dir)
    assert q["blocked"] is False
    assert {g["status"] for g in q["gates"]} == {PASS}


def test_oct_2_shape_discloses_single_source_and_nothing_blocks(settings, classifier, out_dir):
    f = FakeFeeds()
    for i in range(11):
        f.add(URL_A, f"{'Up' if i % 2 else 'Down'} a story {i}", utc(f"2026-10-02T{8 + i:02d}:00:00-04:00"))
    for i in range(4):
        f.add(URL_B, f"{'Up' if i % 2 else 'Down'} b story {i}", utc(f"2026-10-02T{8 + i:02d}:30:00-04:00"))
    r = run(settings, f, classifier, out_dir)
    assert r.result.day.n_headlines == 15
    text = r.path.read_text(encoding="utf-8")
    assert "_Data notes: 73% of headlines from feed-a._" in text
    assert "Gates: 2/2 passed · 1 data note · Not financial advice." in text
    statuses = {g["name"]: g["status"] for g in quality(out_dir)["gates"]}
    assert statuses == {**dict.fromkeys(GATE_ORDER, PASS), "single_source": FIRED}


def test_thin_day_is_published_with_its_n(settings, classifier, out_dir):
    r = run(settings, balanced_feeds(per_feed=3), classifier, out_dir)
    assert "_Data notes: Thin day: 6 headlines._" in r.path.read_text(encoding="utf-8")


def test_stale_pipeline_blocks_writes_quality_json_and_no_report(settings, classifier, out_dir, capsys):
    f = balanced_feeds()
    run(settings, f, classifier, out_dir)  # Oct 2: stored, fresh
    (out_dir / "2026-10-02.md").unlink()

    # Two days later every feed is down, so nothing new is stored.
    f.failing |= {URL_A, URL_B}
    later = NOW + 40 * H
    with pytest.raises(pipeline.GateBlocked, match="blocked by gate pipeline_fresh") as exc:
        run(settings, f, classifier, out_dir, now=later)
    assert not (out_dir / "2026-10-04.md").exists()
    assert exc.value.quality_path == out_dir / "quality" / "2026-10-04.json"
    q = quality(out_dir, "2026-10-04")
    assert q["blocked"] is True
    statuses = {g["name"]: g["status"] for g in q["gates"]}
    assert statuses["pipeline_fresh"] == FAIL
    assert statuses["feeds_down"] == FIRED  # feed gates still evaluated: they explain the block
    assert statuses["thin_day"] == NOT_APPLICABLE  # day gates need the index, which a block never writes
    with closing(store.connect(settings.store.db_path)) as conn:
        assert store.get_daily_index(conn, "2026-10-04") is None


def test_unscored_rows_block_before_a_past_day_is_frozen(settings, classifier, out_dir, monkeypatch):
    # A scoring failure on a backfill. Sep 20 holds one scored row (from an
    # earlier run; outside that run's lede lookback, so not yet written) and
    # one that scoring failed on. Had the index been computed first, Sep 20
    # would be written once from the scored row alone and frozen.
    f = balanced_feeds()
    f.add(URL_A, "Up older story", utc("2026-09-20T10:00:00-04:00"))
    run(settings, f, classifier, out_dir)
    f.add(URL_B, "Down older story", utc("2026-09-20T11:00:00-04:00"))
    monkeypatch.setattr(scoring, "score_store", lambda *a, **kw: 0)
    with pytest.raises(pipeline.GateBlocked, match="all_scored"):
        run(settings, f, classifier, out_dir, date_arg="2026-09-20")
    assert not (out_dir / "2026-09-20.md").exists()
    statuses = {g["name"]: g["status"] for g in quality(out_dir, "2026-09-20")["gates"]}
    assert statuses["all_scored"] == FAIL
    assert statuses["pipeline_fresh"] == NOT_APPLICABLE
    with closing(store.connect(settings.store.db_path)) as conn:
        assert store.get_daily_index(conn, "2026-09-20") is None


def test_past_date_with_an_old_store_is_not_blocked(settings, classifier, out_dir):
    f = balanced_feeds()
    f.add(URL_A, "Up older story", utc("2026-09-28T10:00:00-04:00"))
    run(settings, f, classifier, out_dir)
    f.failing |= {URL_A, URL_B}
    r = run(settings, f, classifier, out_dir, now=NOW + 10 * 24 * H, date_arg="2026-09-28")
    assert r.path.exists()
    assert "Gates: 1/1 passed, 1 not applicable" in r.path.read_text(encoding="utf-8")


def test_cli_exits_non_zero_and_names_the_gate(settings, classifier, out_dir, monkeypatch):
    f = balanced_feeds()
    f.failing |= {URL_A, URL_B}  # nothing ever stored: pipeline_fresh fails
    real = pipeline.run_report
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        pipeline, "run_report",
        lambda s, **kw: real(s, fetch=f, get_classifier=lambda: classifier, now=lambda: NOW, out_dir=out_dir, **kw),
    )
    result = CliRunner().invoke(cli.app, ["report", "--offline"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "error: blocked by gate pipeline_fresh" in result.stderr
    assert "quality/2026-10-02.json" in result.stderr
    assert not (out_dir / "2026-10-02.md").exists()
    assert (out_dir / "quality" / "2026-10-02.json").exists()


def test_t44_env_override_trips_pipeline_fresh_even_with_fresh_rows(tmp_path, classifier, out_dir, monkeypatch):
    # T4.4's forced-failure check. This run stores new rows (0 h old), and
    # 0 >= 0 still fails.
    monkeypatch.setenv("NEWSMOOD_GATES__MAX_FEED_STALENESS_HOURS", "0")
    base = get_settings()
    assert base.gates.max_feed_staleness_hours == 0
    ingest = base.ingest.model_copy(
        update={"delay_seconds": 0, "feeds": [FeedSettings(name="feed-a", url=URL_A), FeedSettings(name="feed-b", url=URL_B)]}
    )
    s = base.model_copy(update={"store": StoreSettings(db_path=str(tmp_path / "t.db")), "ingest": ingest})
    with pytest.raises(pipeline.GateBlocked) as exc:
        run(s, balanced_feeds(), classifier, out_dir)
    assert [r.name for r in exc.value.report.failed] == ["pipeline_fresh"]
    assert not (out_dir / "2026-10-02.md").exists()


def test_rerun_report_and_quality_json_identical_apart_from_run_at(settings, classifier, out_dir):
    f = balanced_feeds()
    first = run(settings, f, classifier, out_dir)
    report_a, q_a = first.path.read_bytes(), quality(out_dir)
    second = run(settings, f, classifier, out_dir, now=NOW + H)
    assert second.path.read_bytes() == report_a
    q_b = quality(out_dir)
    assert q_a["run_at"] != q_b["run_at"]
    assert {**q_a, "run_at": None} == {**q_b, "run_at": None}


# --- store queries behind the block gates ---


def test_store_last_fetched_and_unscored_between(settings, classifier, out_dir):
    with closing(store.connect(settings.store.db_path)) as conn:
        assert store.last_fetched_at(conn) is None
    f = balanced_feeds()
    run(settings, f, classifier, out_dir)
    f.add(URL_B, "Up brand new", utc("2026-10-02T13:30:00-04:00"))
    later = NOW + H
    results = feeds.fetch_all(settings.ingest, fetch=f, now=lambda: later)
    with closing(store.connect(settings.store.db_path)) as conn:
        store.upsert_headlines(conn, (h for r in results for h in r.headlines))
        assert store.last_fetched_at(conn) == later
        assert store.unscored_between(conn, utc("2026-10-02T00:00:00-04:00"), utc("2026-10-03T00:00:00-04:00")) == 1
        assert store.unscored_between(conn, NOW - 30 * 24 * H, NOW - 29 * 24 * H) == 0
