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
    text: str = typer.Option(None, "--text", help="One headline to classify."),
):
    """Classify a headline with the fine-tuned DistilBERT model from the Hub."""
    if text is None:
        raise typer.BadParameter("pass --text; --file and scoring the SQLite store arrive in T2.5")
    # Imported here: torch/transformers live behind the [train] extra.
    from newsmood.models.classifier import classify, load_classifier
    from newsmood.models.loader import ensure_model
    from newsmood.training import finetune

    settings = get_settings()
    device = finetune.resolve_device("auto")
    model, tokenizer = load_classifier(str(ensure_model(settings.model)))
    label, confidence = classify([text], model, tokenizer, device, settings.training.max_length)[0]
    print(f"{label} {confidence:.4f}")


@app.command()
def report():
    """Aggregate scored headlines into a daily markdown report."""
    print("not implemented")


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
    # Imported here, not at module top: torch/transformers live behind the
    # [train] extra, and `newsmood ingest` must work without them.
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
