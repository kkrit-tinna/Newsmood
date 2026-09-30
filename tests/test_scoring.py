"""`newsmood score` (T2.5) with a fake classifier: no model download, no torch
forward pass. The SQLite tests use a real store in a tmp file."""

from __future__ import annotations

import json
from contextlib import closing
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from newsmood.cli import app
from newsmood.data import store
from newsmood.data.feeds import Headline, normalize_title, url_hash
from newsmood.models import scoring

FETCHED = datetime(2026, 9, 29, 1, 0, tzinfo=timezone.utc)
LABELS = ["negative", "neutral", "positive"]


class FakeClassifier:
    """Label and confidence are a function of the text alone, so any
    reordering or row mix-up changes the output. Records each call."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [(LABELS[len(t) % 3], 0.5 + (len(t) % 7) / 20) for t in texts]


def _headline(i: int) -> Headline:
    url = f"https://outlet.test/story-{i}"
    title = f"Story {i} " + "x" * i
    return Headline(url_hash(url), "outlet", title, normalize_title(title), "", url, None, FETCHED)


def _rows(conn):
    return {r["id"]: dict(r) for r in conn.execute("SELECT id, title, label, confidence, scored_at FROM headlines")}


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "newsmood.db"
    with closing(store.connect(path)) as conn:
        store.upsert_headlines(conn, [_headline(i) for i in range(33)])
    return path


# --- SQLite mode ---


def test_score_store_sets_all_three_fields_on_null_rows_only(db):
    with closing(store.connect(db)) as conn:
        pre = _headline(0).id
        conn.execute(
            "UPDATE headlines SET label='negative', confidence=0.99, scored_at='2026-09-28T00:00:00.000000+00:00' WHERE id=?",
            (pre,),
        )
        conn.commit()
        fake = FakeClassifier()

        n = scoring.score_store(conn, lambda: fake, batch_size=32)

        rows = _rows(conn)
    assert n == 32
    assert (rows[pre]["label"], rows[pre]["confidence"], rows[pre]["scored_at"]) == (
        "negative", 0.99, "2026-09-28T00:00:00.000000+00:00"
    )
    assert all(t != rows[pre]["title"] for call in fake.calls for t in call)
    for id_, row in rows.items():
        if id_ == pre:
            continue
        assert (row["label"], row["confidence"]) == fake([row["title"]])[0]
        assert len(row["scored_at"]) == 32 and row["scored_at"].endswith("+00:00")


def test_score_store_writes_in_batches_and_keeps_rows_matched_across_the_boundary(db):
    fake = FakeClassifier()
    with closing(store.connect(db)) as conn:
        assert scoring.score_store(conn, lambda: fake, batch_size=32) == 33
        rows = _rows(conn)
    assert [len(c) for c in fake.calls] == [32, 1]
    assert all((r["label"], r["confidence"]) == fake([r["title"]])[0] for r in rows.values())


def test_score_store_is_idempotent_and_loads_no_model_when_nothing_is_new(db):
    with closing(store.connect(db)) as conn:
        scoring.score_store(conn, FakeClassifier, batch_size=32)
        before = _rows(conn)

        def no_model():
            raise AssertionError("loaded a model with nothing to score")

        assert scoring.score_store(conn, no_model, batch_size=32) == 0
        assert _rows(conn) == before


def test_write_scores_never_overwrites_a_scored_row(db):
    id_ = _headline(1).id
    with closing(store.connect(db)) as conn:
        assert store.write_scores(conn, [(id_, "positive", 0.8)], FETCHED) == 1
        assert store.write_scores(conn, [(id_, "negative", 0.6)], FETCHED) == 0
        assert _rows(conn)[id_]["label"] == "positive"


# --- JSONL ---


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('{"sentence": "a", "label": "neutral"}\n\n   \n{"sentence": "b"}\n')
    assert [r["sentence"] for r in scoring.read_jsonl(path)] == ["a", "b"]


@pytest.mark.parametrize(
    "bad, message",
    [
        ('{"sentence": "a"', "not valid JSON"),
        ('{"text": "a"}', 'missing "sentence"'),
        ('["a"]', 'missing "sentence"'),
        ('{"sentence": 3}', '"sentence" is not a string'),
    ],
)
def test_read_jsonl_names_the_bad_line(tmp_path, bad, message):
    path = tmp_path / "in.jsonl"
    path.write_text('{"sentence": "ok"}\n\n' + bad + "\n")
    with pytest.raises(scoring.InputFileError, match=rf"in\.jsonl:3: {message}"):
        scoring.read_jsonl(path)


def test_score_records_keeps_the_gold_label_and_adds_pred_label():
    records = [{"sentence": "abc", "label": "positive", "id": 7}]
    out = scoring.score_records(records, FakeClassifier())
    assert out == [{"sentence": "abc", "label": "positive", "id": 7, "pred_label": "negative", "confidence": 0.65}]
    assert records == [{"sentence": "abc", "label": "positive", "id": 7}]


# --- benchmark ---


def test_time_batches_excludes_one_warmup_batch_and_times_every_headline():
    from newsmood.models.benchmark import time_batches

    fake = FakeClassifier()
    texts = [str(i) for i in range(33)]
    total, per_headline = time_batches(texts, fake, batch_size=32)
    assert [len(c) for c in fake.calls] == [32, 32, 1]  # warmup, then the whole file
    assert len(per_headline) == 33
    assert total >= 0 and all(t >= 0 for t in per_headline)


# --- CLI ---


@pytest.fixture
def cli(db, monkeypatch):
    """Runs `newsmood score` against the tmp db with the fake classifier."""
    monkeypatch.setenv("NEWSMOOD_STORE__DB_PATH", str(db))
    monkeypatch.setattr(scoring, "resolve_device", lambda choice: "cpu")
    monkeypatch.setattr(scoring, "load_batch_classifier", lambda settings, device: FakeClassifier())
    runner = CliRunner()
    return lambda *args: runner.invoke(app, ["score", *args])


def _unscored(db) -> int:
    with closing(store.connect(db)) as conn:
        return len(store.unscored(conn))


def test_cli_bare_scores_the_store_then_scores_zero(cli, db):
    first = cli()
    assert first.exit_code == 0, first.output
    assert first.stdout == "scored 33 headlines\n"
    assert _unscored(db) == 0
    assert cli().stdout == "scored 0 headlines\n"


def test_cli_text_prints_label_and_writes_nothing(cli, db):
    result = cli("--text", "Profit fell")
    assert result.exit_code == 0, result.output
    assert result.stdout == "positive 0.7000\n"  # 11 chars -> LABELS[2], 0.5 + 4/20
    assert _unscored(db) == 33


def test_cli_file_prints_one_json_line_per_input_and_writes_nothing(cli, db, tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('{"sentence": "abc", "label": "positive"}\n\n{"sentence": "abcd", "label": "neutral"}\n')

    result = cli("--file", str(path))

    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.stdout.splitlines()]
    assert [(r["label"], r["pred_label"]) for r in lines] == [("positive", "negative"), ("neutral", "neutral")]
    assert _unscored(db) == 33


def test_cli_file_error_names_the_line(cli, tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('{"sentence": "ok"}\n{"label": "neutral"}\n')
    result = cli("--file", str(path))
    assert result.exit_code != 0
    assert 'in.jsonl:2: missing "sentence"' in result.output


@pytest.mark.parametrize(
    "args, message",
    [
        (["--text", "a", "--file", "x.jsonl"], "not both"),
        (["--benchmark"], "--benchmark needs --file"),
        (["--text", "a", "--benchmark"], "--benchmark needs --file"),
    ],
)
def test_cli_rejects_invalid_combinations_before_touching_anything(cli, db, args, message):
    result = cli(*args)
    assert result.exit_code != 0
    assert message in result.output
    assert _unscored(db) == 33
