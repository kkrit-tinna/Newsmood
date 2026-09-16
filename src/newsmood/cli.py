import typer

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
def eval_():
    """Evaluate a model or baseline against the test split."""
    print("not implemented")


if __name__ == "__main__":
    app()
