import json
from contextlib import closing
from enum import Enum
from pathlib import Path

import typer

from newsmood.baselines import vader
from newsmood.config import get_settings
from newsmood.data import feeds, store
from newsmood.data.phrasebank import LABEL_NAMES
from newsmood.evaluation.baselines_doc import update_baselines_doc
from newsmood.evaluation.metrics import evaluate, format_eval_result
from newsmood.evaluation.suite import HUMAN_CEILING_NOTE, run_reference_and_baselines

BASELINES_DOC_PATH = Path("docs/baselines.md")

app = typer.Typer()


class Device(str, Enum):
    # Mirrors finetune.DEVICE_CHOICES; listed here so the training stack is
    # imported only when `train` runs. tests/test_training.py keeps them equal.
    auto = "auto"
    cpu = "cpu"
    mps = "mps"
    cuda = "cuda"


@app.command()
def ingest(
    dry_run: bool = typer.Option(False, "--dry-run", help="Fetch and normalize, print per-feed counts, write nothing."),
):
    """Fetch RSS feeds, normalize, and dedupe into SQLite."""
    settings = get_settings()
    results = feeds.fetch_all(settings.ingest)
    print(feeds.format_dry_run(results))
    if dry_run:
        return
    with closing(store.connect(settings.store.db_path)) as conn:
        result = store.upsert_headlines(conn, (h for r in results for h in r.headlines))
    print(f"{settings.store.db_path}: {result.summary()}")


@app.command()
def score(
    text: str = typer.Option(None, "--text", help="One headline to classify. Prints label and confidence; writes nothing."),
    file: Path = typer.Option(
        None, "--file", help='JSONL with a "sentence" per line. Prints each line plus pred_label/confidence; writes nothing.'
    ),
    benchmark: bool = typer.Option(False, "--benchmark", help="With --file: print throughput and latency instead."),
    device: Device = typer.Option(None, "--device", help="Default: cpu with --benchmark, otherwise auto (cuda > mps > cpu)."),
):
    """Classify headlines with the fine-tuned model. With no options, score unscored rows in SQLite."""
    if text is not None and file is not None:
        raise typer.BadParameter("pass --text or --file, not both")
    if benchmark and file is None:
        raise typer.BadParameter("--benchmark needs --file")
    
    # Imported here, not at module top: importing torch takes seconds, and
    # `newsmood ingest` should not pay for it.
    from newsmood.models import scoring

    settings = get_settings()
    choice = device.value if device is not None else ("cpu" if benchmark else "auto")
    try:
        resolved = scoring.resolve_device(choice)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--device")

    if text is not None:
        label, confidence = scoring.load_batch_classifier(settings, resolved)([text])[0]
        print(f"{label} {confidence:.4f}")
        return

    if file is not None:
        try:
            records = scoring.read_jsonl(file)
        except (OSError, scoring.InputFileError) as e:
            raise typer.BadParameter(str(e), param_hint="--file")
        classify = scoring.load_batch_classifier(settings, resolved)
        if benchmark:
            from newsmood.models.benchmark import format_benchmark, run_benchmark

            texts = [r["sentence"] for r in records]
            print(format_benchmark(run_benchmark(texts, classify, settings.model.batch_size, str(resolved))))
            return
        for record in scoring.score_records(records, classify):
            print(json.dumps(record, ensure_ascii=False))
        return

    with closing(store.connect(settings.store.db_path)) as conn:
        n = scoring.score_store(conn, lambda: scoring.load_batch_classifier(settings, resolved), settings.model.batch_size)
    print(f"scored {n} headlines")


@app.command()
def report(
    date: str = typer.Option(None, "--date", help="ET calendar day, YYYY-MM-DD. Default: today in America/New_York."),
    offline: bool = typer.Option(True, "--offline/--online", help="Offline is the default and the only mode until T4.1."),
):
    """Ingest, score, index and write reports/{date}.md. Prints the report's path.
    Exits 1, naming the gate on stderr, when a quality gate blocks the run."""
    # Imported here, not at module top: scoring imports torch, and
    # `newsmood ingest` should not pay for it.
    from newsmood.reporting import pipeline

    try:
        run = pipeline.run_report(get_settings(), date_arg=date, offline=offline)
    except pipeline.ReportError as e:
        raise typer.BadParameter(str(e))
    except pipeline.GateBlocked as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    print(run.path)


@app.command()
def train(
    dry_run: bool = typer.Option(False, "--dry-run", help="One forward and one backward pass, save nothing."),
    limit: int = typer.Option(None, "--limit", help="Training rows in the dry-run batch (default 32). Dry run only."),
    device: Device = typer.Option(
        Device.auto,
        "--device",
        help="auto picks cuda > mps > cpu. If MPS looks stuck on small batches, try cpu.",
    ),
):
    """Fine-tune DistilBERT on the Financial PhraseBank."""
    # Imported here, not at module top: importing torch takes seconds, and
    # `newsmood ingest` should not pay for it.
    from newsmood.training import finetune

    if limit is not None and not dry_run:
        raise typer.BadParameter("--limit applies to --dry-run only; the fine-tune uses the whole train split")
    try:
        resolved = finetune.resolve_device(device.value)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--device")
    settings = get_settings()
    print(f"device: {resolved} (--device {device.value}), seed {settings.dataset.seed}")
    if dry_run:
        print(finetune.dry_run(settings, 32 if limit is None else limit, resolved).summary())
        return
    finetune.fine_tune(settings, resolved)


@app.command(name="eval")
def eval_(
    model: str = typer.Option(None, "--model", help="Baseline or model to evaluate, e.g. 'vader'."),
    all_: bool = typer.Option(False, "--all", help="Run every reference row and baseline, write docs/baselines.md."),
):
    """Evaluate a model or baseline against the test split."""
    settings = get_settings()

    if all_:
        table = run_reference_and_baselines(settings)
        update_baselines_doc(BASELINES_DOC_PATH, table, HUMAN_CEILING_NOTE, settings.dataset.seed)
        print(f"wrote {BASELINES_DOC_PATH} ({len(table)} methods)")
        return

    if model is None:
        raise typer.BadParameter("pass --model <name> or --all")
    if model == "vader":
        _, y_true, y_pred = vader.predict_test_split(settings)
    else:
        raise typer.BadParameter(f"unknown model {model!r}, expected one of: vader")

    result = evaluate(y_true, y_pred, LABEL_NAMES)
    print(format_eval_result(model, result))


if __name__ == "__main__":
    app()
