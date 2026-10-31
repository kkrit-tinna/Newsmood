# newsmood

A CLI that labels financial RSS headlines with a fine-tuned DistilBERT and writes a daily market-mood report from computed evidence.

**Sample report:** [reports/2026-10-30.md](reports/2026-10-30.md). A new one is committed each weekday.

Not financial advice. See the [model card](docs/model_card.md).

## Try it

```
git clone https://github.com/kkrit-tinna/Newsmood.git && cd Newsmood
python -m venv .venv && source .venv/bin/activate
make demo
```

Needs network and no API key. The first run downloads the 269 MB model. It runs the real pipeline on live RSS against a temp DB and output folder, then prints where the report is. A fresh DB is a thin day, so expect a data note and possibly no clusters.

## Results

Test split of Financial PhraseBank (`sentences_75agree`), n=518.

| Method | Accuracy | Macro-F1 | Negative recall |
|---|---|---|---|
| Majority class (floor) | 62.16% | 25.56% | 0.00% |
| VADER | 54.05% | 45.83% | 22.22% |
| TF-IDF + Logistic Regression | 83.59% | 79.97% | 77.78% |
| **DistilBERT (fine-tuned)** | **92.08%** | **90.19%** | **96.83%** |
| FinBERT (trained on PhraseBank, reference only) | 95.37% | 94.45% | 98.41% |

FinBERT has probably seen these test sentences in training, so it is neither zero-shot nor a ceiling. Full analysis: [docs/baselines.md](docs/baselines.md).

## The number that matters

On 100 live headlines labeled by hand, accuracy is **59.00%** (Wilson 95% [49.20, 68.13]), against 92.08% on PhraseBank. Positive recall is 25.93%. The reference is one labeler, who agreed with themselves on 70.00% of a 20-headline blind relabel. Treat single labels as noisy. Details: [docs/drift.md](docs/drift.md).

## Install

Python 3.11+. From a clone:

```
pip install .
```

Or install the wheel attached to the [v0.1.0 GitHub release](https://github.com/kkrit-tinna/Newsmood/releases/tag/v0.1.0) (tested on Python 3.12.5, macOS 14.6):

```
pip install newsmood-0.1.0-py3-none-any.whl
newsmood report --offline
```

`data/newsmood.db` and `reports/` are created relative to the working directory.

## How it works

**ingest → score → index → gates → report**

1. **Ingest:** fetch RSS feeds and dedupe on the normalized title.
2. **Score:** DistilBERT labels the raw title, at a pinned model revision.
3. **Index:** Python computes the mood, `100 × Σ(confidence × sign) / Σ confidence`.
4. **Gates:** blocking gates stop a broken run; disclosure gates add a data note to a thin one. Blocks run before the index is computed.
5. **Report:** Markdown, with the lede and evidence blocks computed in Python. No LLM or API key is involved.

GitHub Actions ingests every 6 h and reports on weekdays at 22:00 UTC. The SQLite store lives on the orphan `data` branch, which is never merged. Each workflow fetches it and pushes it back.

## Using the data

```
git fetch origin data && git show origin/data:newsmood.db > data/newsmood.db
```

- `headlines`: `id, source, title, title_norm, summary, url, published_at, fetched_at, label, confidence, scored_at`
- `daily_index`: `date, mood_index, n_headlines, n_positive, n_neutral, n_negative, mean_confidence, n_low_conf, n_sources, source_counts, terms, model_id, computed_at`

`source_counts` and `terms` are JSON text. `model_id` records the model behind each day's index.

```python
import sqlite3
import pandas as pd

pd.read_sql("SELECT * FROM daily_index", sqlite3.connect("data/newsmood.db"))
```

## When a run fails

- **Exit 1: a gate blocked the run.** The gate is named on stderr, no report or index row is written, and `reports/quality/{date}.json` records why. An uncaught exception also exits 1; only a gate block prints `blocked by gate` and sets `"blocked": true` in that file.
- **Exit 2: rejected request.** A malformed or future `--date`, or `--online`, which isn't implemented. Nothing runs.
- **Quality JSON stays local.** It is written by every run that reaches the gates, and it is never committed. On Actions the job summary is the record.
- **Backfill a missed day:** `newsmood report --offline --date YYYY-MM-DD`. Feed gates are `not_applicable` on a past date, and a past day's index row is frozen once written.
- **Restore the DB:** `git fetch origin data && git show origin/data:newsmood.db > data/newsmood.db`.

## Engineering notes

- **MPS:** `newsmood report` scores on CPU by default (`model.device`), because MPS aborts on some torch/macOS combinations. Set `NEWSMOOD_MODEL__DEVICE=mps` or `auto` to opt in, with `PYTORCH_ENABLE_MPS_FALLBACK=1` exported so unsupported ops fall back to CPU. `newsmood score` and `newsmood train` have their own `--device`, newsmood score defaults to --device auto, if it aborts on Apple Silicon, pass --device cpu.
- **Reports render fully only on GitHub:** they use a Mermaid pie and `<details>`.
- **The cluster block declines rather than manufacture a theme:** on low-volume days, and on busier days when no tight cluster forms, as in the sample report. On a store with under 30 days of history, cluster names rank by raw count.
- **`cnbc-finance` has been stale since Oct 21:** the feed responds with nothing new, so reports carry a data note and count two sources.

## Predecessor

[social_media_sentiment_analysis](https://github.com/kkrit-tinna/social_media_sentiment_analysis) stalled on a synthetic Kaggle dataset with unusable sentiment labels; newsmood uses a real corpus with three clean classes and ships as an installable tool instead of a notebook.
