"""End-to-end report pipeline (T4.3): temp DB, fake feeds, fake classifier,
fixed clock, temp output dir. No network, no model, no real reports/."""

from __future__ import annotations

from contextlib import closing
from datetime import date, datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

import pytest
from typer.testing import CliRunner

from newsmood import cli
from newsmood.config import FeedSettings, StoreSettings, get_settings
from newsmood.data import store
from newsmood.reporting import pipeline
from newsmood.reporting.aggregate import compute_days, day_bounds
from newsmood.reporting.explain import build_explanation
from newsmood.reporting.report_data import build_report_data
from newsmood.reporting.template import render

URL_A = "https://a.test/rss"
URL_B = "https://b.test/rss"
# 14:00 ET on Fri Oct 2, 2026.
NOW = datetime(2026, 10, 2, 18, 0, tzinfo=timezone.utc)


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(timezone.utc)


class FakeFeeds:
    """Serves RSS bodies per url. Items are (title, link, published_at)."""

    def __init__(self):
        self.items: dict[str, list[tuple[str, str, datetime]]] = {URL_A: [], URL_B: []}
        self.failing: set[str] = set()
        self.calls: list[str] = []

    def add(self, url: str, title: str, published_at: datetime) -> None:
        slug = title.lower().replace(" ", "-")
        self.items[url].append((title, f"{url.removesuffix('/rss')}/{slug}", published_at))

    def __call__(self, url: str, user_agent: str, timeout: float) -> bytes:
        self.calls.append(url)
        if url in self.failing:
            raise OSError(f"connection refused: {url}")
        items = "".join(
            f"<item><title>{escape(t)}</title><link>{escape(link)}</link>"
            f"<pubDate>{format_datetime(p)}</pubDate></item>"
            for t, link, p in self.items[url]
        )
        return f'<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>{items}</channel></rss>'.encode()


class FakeClassifier:
    """"Up ..." → positive, "Down ..." → negative, else neutral. Counts calls."""

    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, texts: list[str]) -> list[tuple[str, float]]:
        self.seen += texts
        out = []
        for t in texts:
            if t.startswith("Up"):
                out.append(("positive", 0.91))
            elif t.startswith("Down"):
                out.append(("negative", 0.83))
            else:
                out.append(("neutral", 0.55))
        return out


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
def fake_feeds():
    f = FakeFeeds()
    # Oct 2 (ET): two from A, one from B. Oct 1 (ET): one late-evening item
    # whose UTC timestamp is already Oct 2.
    f.add(URL_A, "Up stocks rally on earnings", utc("2026-10-02T09:30:00-04:00"))
    f.add(URL_A, "Fed holds rates steady", utc("2026-10-02T11:00:00-04:00"))
    f.add(URL_B, "Down oil slides on supply", utc("2026-10-02T12:15:00-04:00"))
    f.add(URL_B, "Up bank shares climb late", utc("2026-10-01T21:00:00-04:00"))
    return f


@pytest.fixture
def classifier():
    return FakeClassifier()


def run(settings, fake_feeds, classifier, out_dir, *, now=NOW, date_arg=None, **kw):
    return pipeline.run_report(
        settings, date_arg=date_arg, fetch=fake_feeds, get_classifier=lambda: classifier,
        now=lambda: now, out_dir=out_dir, **kw,
    )


def expected_report(settings, day: date, now=NOW) -> str:
    """What the report must be, rebuilt from the store by the T4.2 functions."""
    tz = ZoneInfo(settings.index.timezone)
    with closing(store.connect(settings.store.db_path)) as conn:
        (result,) = compute_days(conn, [day], settings, now=now)
        rows = store.scored_headlines_between(conn, *day_bounds(day, tz))
        data = build_report_data(result, rows, settings)
        lookup, _ = pipeline.session_lookup(conn, settings, now, (result,))
        themes = pipeline.theme_inputs(conn, settings, result, rows)
        return render(data, build_explanation(data, settings, lookup=lookup, themes=themes))


def daily_row(settings, day: str):
    with closing(store.connect(settings.store.db_path)) as conn:
        return store.get_daily_index(conn, day)


# --- the full run ---


def test_full_run_writes_report_matching_render_of_build(settings, fake_feeds, classifier, out_dir):
    r = run(settings, fake_feeds, classifier, out_dir)
    assert r.path == out_dir / "2026-10-02.md"
    text = r.path.read_text(encoding="utf-8")
    assert text == expected_report(settings, date(2026, 10, 2))
    assert r.result.day.n_headlines == 3
    assert "Up stocks rally on earnings" in text
    assert "Up bank shares climb late" not in text  # Oct 1 in ET
    assert sorted(fake_feeds.calls) == [URL_A, URL_B]
    assert len(classifier.seen) == 4
    assert [p.name for p in out_dir.iterdir()] == ["2026-10-02.md"]  # no temp file left


def test_output_dir_defaults_to_report_output_dir(settings, fake_feeds, classifier, tmp_path):
    configured = tmp_path / "from-config"
    settings = settings.model_copy(update={"report": settings.report.model_copy(update={"output_dir": str(configured)})})
    r = pipeline.run_report(settings, fetch=fake_feeds, get_classifier=lambda: classifier, now=lambda: NOW)
    assert r.path == configured / "2026-10-02.md"
    assert r.path.exists()


def test_rerun_with_no_new_headlines_is_byte_identical(settings, fake_feeds, classifier, out_dir):
    first = run(settings, fake_feeds, classifier, out_dir).path.read_bytes()
    later_same_day = datetime(2026, 10, 2, 23, 0, tzinfo=timezone.utc)
    second = run(settings, fake_feeds, classifier, out_dir, now=later_same_day).path.read_bytes()
    assert first == second


def test_rerun_overwrites_existing_report(settings, fake_feeds, classifier, out_dir):
    out_dir.mkdir()
    (out_dir / "2026-10-02.md").write_text("stale\n")
    path = run(settings, fake_feeds, classifier, out_dir).path
    assert path.read_text(encoding="utf-8") == expected_report(settings, date(2026, 10, 2))


def test_late_headline_for_frozen_day_keeps_frozen_mood_and_notes_it(settings, fake_feeds, classifier, out_dir):
    old = "2026-09-28"
    fake_feeds.add(URL_A, "Up chipmakers gain", utc("2026-09-28T10:00:00-04:00"))
    fake_feeds.add(URL_B, "Down retail sales miss", utc("2026-09-28T11:00:00-04:00"))
    first = run(settings, fake_feeds, classifier, out_dir, date_arg=old)
    assert first.result.action == "written"
    frozen = daily_row(settings, old)

    fake_feeds.add(URL_B, "Up late rally in tech", utc("2026-09-28T15:00:00-04:00"))
    second = run(settings, fake_feeds, classifier, out_dir, date_arg=old)
    assert second.result.action == "kept"
    assert daily_row(settings, old)["mood_index"] == frozen["mood_index"]
    assert daily_row(settings, old)["terms"] == frozen["terms"]

    text = second.path.read_text(encoding="utf-8")
    assert second.result.day.mood == frozen["mood_index"]
    header = f"**Mood index: {frozen['mood_index']:+.1f}**"
    assert header in text
    assert "_1 later headline for this date arrived after its index was frozen and is not counted._" in text
    assert "Up late rally in tech" not in text
    assert "2 headlines" in text


def test_failed_write_leaves_previous_report_and_no_temp_file(tmp_path, monkeypatch):
    path = tmp_path / "2026-10-02.md"
    path.write_text("previous\n")

    def crash(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline.os, "replace", crash)
    with pytest.raises(OSError, match="disk full"):
        pipeline.write_atomic(path, "new\n")
    assert path.read_text() == "previous\n"
    assert [p.name for p in tmp_path.iterdir()] == ["2026-10-02.md"]


def test_full_row_query_matches_index_query_membership(settings, fake_feeds, classifier, out_dir):
    run(settings, fake_feeds, classifier, out_dir)
    bounds = day_bounds(date(2026, 10, 2), ZoneInfo(settings.index.timezone))
    with closing(store.connect(settings.store.db_path)) as conn:
        full = store.scored_headlines_between(conn, *bounds)
        index_rows = store.scored_rows_between(conn, *bounds)
    assert [r["id"] for r in full] == [r["id"] for r in index_rows]
    assert set(full[0]) == {"id", "title", "url", "source", "published_at", "label", "confidence"}


# --- dates ---


def test_default_date_is_et_not_utc(settings, fake_feeds, classifier, out_dir):
    # 01:30 UTC on Oct 3 is 21:30 ET on Oct 2.
    r = run(settings, fake_feeds, classifier, out_dir, now=datetime(2026, 10, 3, 1, 30, tzinfo=timezone.utc))
    assert r.path.name == "2026-10-02.md"
    assert not (out_dir / "2026-10-03.md").exists()


def test_today_also_recomputes_yesterday(settings, fake_feeds, classifier, out_dir):
    r = run(settings, fake_feeds, classifier, out_dir)
    assert [(d.date.isoformat(), d.action) for d in r.computed] == [("2026-10-01", "recomputed"), ("2026-10-02", "recomputed")]
    assert daily_row(settings, "2026-10-01")["n_headlines"] == 1

    # A late headline for yesterday arrives; the next run for today updates it.
    fake_feeds.add(URL_A, "Down late selloff", utc("2026-10-01T23:00:00-04:00"))
    run(settings, fake_feeds, classifier, out_dir)
    assert daily_row(settings, "2026-10-01")["n_headlines"] == 2


def test_past_date_computes_only_that_day(settings, fake_feeds, classifier, out_dir):
    fake_feeds.add(URL_A, "Up older story", utc("2026-09-29T10:00:00-04:00"))
    r = run(settings, fake_feeds, classifier, out_dir, date_arg="2026-09-29")
    assert [d.date.isoformat() for d in r.computed] == ["2026-09-29"]
    assert daily_row(settings, "2026-09-28") is None
    assert daily_row(settings, "2026-10-02") is None


@pytest.mark.parametrize("bad", ["2026-10-03", "2027-01-01"])
def test_future_date_rejected_before_any_work(settings, fake_feeds, classifier, out_dir, bad):
    with pytest.raises(pipeline.ReportError, match="in the future"):
        run(settings, fake_feeds, classifier, out_dir, date_arg=bad)
    assert fake_feeds.calls == []
    assert not out_dir.exists()
    assert not Path(settings.store.db_path).exists()


@pytest.mark.parametrize("bad", ["2026-13-01", "2026-02-30", "10/02/2026", "20261002", "2026-10-2", "yesterday", ""])
def test_malformed_date_rejected_before_any_work(settings, fake_feeds, classifier, out_dir, bad):
    with pytest.raises(pipeline.ReportError, match="--date"):
        run(settings, fake_feeds, classifier, out_dir, date_arg=bad)
    assert fake_feeds.calls == []
    assert not out_dir.exists()


# --- feed failures and empty days ---


def test_one_feed_failing_still_writes_report(settings, fake_feeds, classifier, out_dir, capsys):
    fake_feeds.failing.add(URL_B)
    r = run(settings, fake_feeds, classifier, out_dir)
    err = capsys.readouterr().err
    assert "warning: feed feed-b failed" in err
    assert "1/2 feeds live" in err
    assert r.result.day.n_headlines == 2  # feed-a's two Oct 2 items
    assert r.path.read_text(encoding="utf-8") == expected_report(settings, date(2026, 10, 2))


def test_all_feeds_failing_builds_report_from_stored_data(settings, fake_feeds, classifier, out_dir, capsys):
    run(settings, fake_feeds, classifier, out_dir)
    before = (out_dir / "2026-10-02.md").read_bytes()
    capsys.readouterr()

    fake_feeds.failing |= {URL_A, URL_B}
    r = run(settings, fake_feeds, classifier, out_dir)
    err = capsys.readouterr().err
    assert "warning: all 2 feeds failed" in err
    assert r.result.day.n_headlines == 3
    assert r.path.read_bytes() == before


def test_empty_day_writes_no_headlines_report(settings, fake_feeds, classifier, out_dir):
    r = run(settings, fake_feeds, classifier, out_dir, date_arg="2026-09-15")
    assert r.result.action == "empty"
    text = r.path.read_text(encoding="utf-8")
    assert "No headlines for this date." in text
    assert "Mood index" not in text
    assert daily_row(settings, "2026-09-15") is None


# --- offline, keys and output streams ---


def test_runs_without_anthropic_key(settings, fake_feeds, classifier, out_dir, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = run(settings, fake_feeds, classifier, out_dir)
    assert r.path.exists()


def test_online_mode_errors_before_any_work(settings, fake_feeds, classifier, out_dir):
    with pytest.raises(pipeline.ReportError, match=r"online mode not yet implemented \(T4\.1\)"):
        run(settings, fake_feeds, classifier, out_dir, offline=False)
    assert fake_feeds.calls == []
    assert not Path(settings.store.db_path).exists()


def test_pipeline_prints_progress_to_stderr_only(settings, fake_feeds, classifier, out_dir, capsys):
    run(settings, fake_feeds, classifier, out_dir)
    out, err = capsys.readouterr()
    assert out == ""
    assert "ingest: 2/2 feeds live" in err
    assert "score: scored 4 headlines" in err
    assert "index: 2026-10-02 recomputed" in err


def test_cli_stdout_is_only_the_path(settings, fake_feeds, classifier, out_dir, monkeypatch):
    real = pipeline.run_report
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        pipeline, "run_report",
        lambda s, **kw: real(s, fetch=fake_feeds, get_classifier=lambda: classifier, now=lambda: NOW, out_dir=out_dir, **kw),
    )
    result = CliRunner().invoke(cli.app, ["report", "--offline"])
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{out_dir / '2026-10-02.md'}\n"
    assert "ingest: 2/2 feeds live" in result.stderr


@pytest.mark.parametrize(
    "args, message",
    [
        (["report", "--online"], "online mode not yet implemented (T4.1)"),
        (["report", "--date", "10/02/2026"], "YYYY-MM-DD"),
        (["report", "--date", "2999-01-01"], "in the future"),
    ],
)
def test_cli_rejects_bad_requests_without_touching_store(settings, monkeypatch, args, message):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    def no_store(*a, **kw):
        raise AssertionError("store opened for a rejected request")

    monkeypatch.setattr(store, "connect", no_store)
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert message in " ".join(result.stderr.split())


# --- previous session for the lede (T4.2b) ---


def test_lede_previous_session_skips_thin_day_and_writes_older_candidate(settings, fake_feeds, classifier, out_dir):
    # Oct 1 holds 1 headline (below prev_session_min_headlines); Sep 30 holds 5.
    for i in range(5):
        fake_feeds.add(URL_A, f"Up story {i}", utc(f"2026-09-30T1{i}:00:00-04:00"))
    r = run(settings, fake_feeds, classifier, out_dir)
    assert [(d.date.isoformat(), d.action) for d in r.lookback] == [("2026-09-30", "written")]
    assert daily_row(settings, "2026-09-30")["n_headlines"] == 5
    text = r.path.read_text(encoding="utf-8")
    assert "96.5" in text and "Wed Sep 30" in text  # +3.5 today vs +100.0 on Sep 30
    assert text == expected_report(settings, date(2026, 10, 2))


def test_lede_lookback_stores_no_empty_days(settings, fake_feeds, classifier, out_dir):
    r = run(settings, fake_feeds, classifier, out_dir)
    # Oct 1 came from this run's own compute; Sep 27-30 have no headlines.
    assert [(d.date.isoformat(), d.action) for d in r.lookback] == [(f"2026-09-{d}", "empty") for d in (30, 29, 28, 27)]
    assert all(daily_row(settings, f"2026-09-{d}") is None for d in (27, 28, 29, 30))
    assert "(index +3.5)" in r.path.read_text(encoding="utf-8")  # no previous session, no change clause


def test_window_mode_report_has_fit_line_and_no_oov_line(settings, fake_feeds, classifier, out_dir):
    text = run(settings, fake_feeds, classifier, out_dir).path.read_text(encoding="utf-8")
    # 3 headlines on Oct 2 plus the Oct 1 one in the background window.
    assert "_Themes use TF-IDF fit on 4 headlines (2026-10-01–2026-10-02)._" in text
    assert "outside it" not in text


def test_phrasebank_oov_advice_goes_to_stderr_not_the_report(settings, fake_feeds, classifier, out_dir, tmp_path, capsys):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer

    artifacts = tmp_path / "baselines"
    artifacts.mkdir()
    joblib.dump(TfidfVectorizer().fit(["profit rose", "sales fell"]), artifacts / "tfidf_vectorizer.joblib")
    settings = settings.model_copy(update={
        "explain": settings.explain.model_copy(update={"vectorizer": "phrasebank"}),
        "logreg": settings.logreg.model_copy(update={"artifact_dir": str(artifacts)}),
    })
    text = run(settings, fake_feeds, classifier, out_dir).path.read_text(encoding="utf-8")
    err = capsys.readouterr().err
    assert "consider explain.vectorizer: window" in err
    assert "PhraseBank baseline's vocabulary: 100.0% of this day's words" in text
    assert "explain.vectorizer" not in text and "Themes use TF-IDF fit" not in text
