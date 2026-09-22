from pathlib import Path

import typer

from newsmood.baselines import vader
from newsmood.config import get_settings
from newsmood.data.phrasebank import LABEL_NAMES
from newsmood.evaluation.baselines_doc import update_baselines_doc
from newsmood.evaluation.metrics import evaluate, format_eval_result
from newsmood.evaluation.suite import HUMAN_CEILING_NOTE, run_reference_and_baselines

BASELINES_DOC_PATH = Path("docs/baselines.md")

app = typer.Typer()


@app.command()
def ingest():
    """Fetch RSS feeds, normalize, and dedupe into SQLite."""
    print("not implemented")


@app.command()
def score():
    """Classify unscored headlines with the fine-tuned DistilBERT model."""
    print("not implemented")


@app.command()
def report():
    """Aggregate scored headlines into a daily markdown report."""
    print("not implemented")


@app.command()
def train():
    """Fine-tune DistilBERT on the Financial PhraseBank."""
    print("not implemented")


@app.command(name="eval")
def eval_(
    model: str = typer.Option(None, "--model", help="Baseline or model to evaluate, e.g. 'vader'."),
    all_: bool = typer.Option(False, "--all", help="Run every reference row and baseline, write docs/baselines.md."),
):
    """Evaluate a model or baseline against the test split."""
    settings = get_settings()

    if all_:
        table = run_reference_and_baselines(settings)
        update_baselines_doc(BASELINES_DOC_PATH, table, HUMAN_CEILING_NOTE)
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
