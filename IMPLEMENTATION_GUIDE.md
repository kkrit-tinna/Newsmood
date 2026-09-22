# Newsmood — Implementation Guide

**Status:** spec for a 5-week build, Sep 16 – Oct 18, 2026 — **offline-first**
**Budget:** 15–30 min/day Mon–Fri, 1–2 hours Sunday
**Audience:** the author, and Claude Code

> **Running in parallel with `movie-rec-system-v2`,** which has the same daily budget again on top of this one. Sunday is the only slot with real room in it, so the three tasks flagged ⚠️ below are placed there deliberately. Do not move them to a weekday.

---

## 0. How to use this document

**Humans:** read §1–§3 once, then work through §4–§7 one task per day.

**Claude Code:** this file is the source of truth. Rules:

1. Do exactly one task per session. A task is one `T#.#` block.
2. Never start a task whose **Depends on** is not yet merged.
3. `CLAUDE.md` holds the standing rules, module boundaries and invariants. Not optional reading.
4. Every task ends with its **Done when** command passing. If it fails, fix it in the same session; do not proceed to the next task.
5. Never call a paid API from a test. Tests mock the Anthropic client. See §3.
6. Never commit `.env`, an API key, a model checkpoint, or anything in `data/`.
7. If reality disagrees with this spec (a config has a different row count, an RSS feed is dead), **update this file in the same commit** and note it under §9 Deviations. Do not silently work around it.

---

---

## 0.5 Schedule

Tasks are listed in §4–§7 in **dependency order**. They are *executed* in the order below, which is not the same thing — ingest (T3.1/T3.2) has no dependency on the model, so it moves early to reach a working report sooner.

**No sessions on Sunday Sep 20 or Sunday Oct 4.**

### Milestone: first offline report — target **Fri Oct 2**, slack to **Fri Oct 9**

| Date | Task | Notes |
|---|---|---|
| Wed Sep 16 | T1.1 skeleton | `.gitignore` first |
| Thu Sep 17 | T1.2 dataset + splits | |
| Fri Sep 18 | T1.3 VADER baseline | |
| ~~Sun Sep 20~~ | — | *off* |
| Mon Sep 21 | T1.4 LogReg baseline | |
| Tue Sep 22 | T1.5 metrics + floor | |
| Wed Sep 23 | T1.6 baseline writeup | |
| Thu Sep 24 | T3.1 RSS ingest | *moved up — no model needed* |
| Fri Sep 25 | T3.2 SQLite store | *moved up* |
| **Sun Sep 27** | **T2.1 + T2.2 + T2.3** | **~2 h. Scaffolding, fine-tune, evaluate** |
| Mon Sep 28 | T2.4 Hub publish | |
| Tue Sep 29 | T2.5 batched scoring | |
| Wed Sep 30 | T3.3 mood index | |
| Thu Oct 1 | T4.2 report template | |
| **Fri Oct 2** | **T4.3 first offline report** | **← milestone** |
| ~~Sun Oct 4~~ | — | *off* |
| Mon Oct 5 | T4.2b Why section, part 1 | contributors + source skew |
| Tue Oct 6 | T4.2b Why section, part 2 | distinctive terms + clusters |
| Wed Oct 7 | T3.5 quality gates | |
| Thu Oct 8 | T2.6 model card | |
| Fri Oct 9 | T4.5 `make demo` | |
| **Sun Oct 11** | **T3.4 drift analysis** | **~90 min, one sitting** |
| Mon Oct 12 | README | |
| Tue Oct 13 | T3.7 runbook | |
| Wed Oct 14 | T4.1 Claude client | *optional — online mode* |
| Thu Oct 15 | online report wiring | *optional* |
| Fri Oct 16 | buffer | |
| **Sun Oct 18** | **T4.4 Actions + T4.6 release** | **~60 min** |

### The two sessions that cannot move

**Sun Sep 27** — the fine-tune needs 20–40 minutes of unattended wall clock plus setup and evaluation. If it lands at the majority-class rate and needs a second attempt, the milestone slips to Oct 9. That is what the Oct 9 buffer is for.

**Sun Oct 11** — T3.4 hand-labels 100 headlines. Do it in one sitting so your own labeling stays consistent; splitting it across two weekdays changes the labels partway through.

### What "offline-first" changes

`--offline` produces the whole report, including the one-sentence template lede (T4.2b, block 0) and every evidence block below it. The only thing it omits is the multi-sentence Claude narrative. That takes T4.1 (the Claude client) off the critical path entirely, which is why it sits on Oct 14, marked optional. Everything up to and including the Oct 2 milestone runs with **no API key and no paid service**.

If you never enable online mode, the project is still complete: `make demo`, the model, the index, the tables, the gates, the drift analysis and the README all stand. You lose one section of prose.

---

## 1. Goal

### User story

> A user installs `newsmood` with pip, runs `newsmood report`, and gets a markdown file for today: financial headlines pulled from public feeds, each classified positive / neutral / negative by a fine-tuned model, an aggregate market-mood score with its history, and a short narrative written by Claude that names the stories driving the number.

### Engineering goal

Ship an installable CLI package that:

- fine-tunes DistilBERT on a real financial corpus and **beats a stated baseline floor**, not an imaginary one
- classifies 1,000+ headlines in under a minute on a laptop CPU, no GPU required
- measures its own out-of-domain degradation instead of assuming the test number holds in production
- runs unattended daily on GitHub Actions for under $1/month in API spend
- can be installed and run end-to-end by a stranger with no API key, in one command

### Explicit non-goals

- **No web service, no cloud infrastructure, no Docker.** This is a CLI tool and a scheduled job. The v1 sibling project (`movie-rec-system-v2`) covers cloud deployment; duplicating it here adds cost and no new interview material.
- **No trading signals, no price prediction, no backtest.** Sentiment of news text is the deliverable. Any claim that it predicts returns needs an event study this project does not have time for, and asserting it without one is the single fastest way to lose credibility with a finance-literate interviewer.
- **No per-ticker attribution.** Headlines mention companies; mapping them to tickers reliably is an NER project. Out of scope.
- **No real-time streaming.** Daily batch. The word "real-time" in the original plan described a scheduler, which is not what real-time means.

---

## 2. Target architecture

```
                     ┌────────────────────────────────────┐
  GitHub Actions     │  newsmood ingest                   │
  cron 22:00 UTC ──► │  RSS feeds → normalize → dedupe    │
  (weekdays)         │  → SQLite  data/newsmood.db        │
                     └──────────────┬─────────────────────┘
                                    ▼
                     ┌────────────────────────────────────┐
                     │  newsmood score                    │
                     │  DistilBERT from HF Hub (cached)   │
                     │  batched CPU inference             │
                     │  → label + confidence per headline │
                     └──────────────┬─────────────────────┘
                                    ▼
                     ┌────────────────────────────────────┐
                     │  newsmood report                   │
                     │  aggregate → mood index (-100..100)│
                     │  quality gates (abort on fail)     │
                     │  → Claude Haiku 4.5 → narrative    │
                     │  → reports/YYYY-MM-DD.md           │
                     └──────────────┬─────────────────────┘
                                    ▼
                   commit report to repo  +  job summary
```

### Why each piece

| Component | Why not something else |
|---|---|
| SQLite | The whole corpus is a few MB of text per month. A hosted database would cost more than the API bill and add a credential to leak |
| RSS via `feedparser` | No key, no quota, no terms-of-service problem, real-time. See §3 Sources for why every API alternative was rejected |
| Model on HF Hub, not in the wheel | A fine-tuned DistilBERT is ~265 MB. PyPI's per-file limit is 100 MB. Lazy-download to a cache dir on first run |
| DistilBERT, not FinBERT | FinBERT is already trained on financial sentiment. Using it means there is no fine-tuning to show. FinBERT appears in this project only as a **zero-shot reference ceiling** in T2.3 |
| Claude Haiku 4.5 | Report generation is summarization of text you already classified. Haiku is $1/$5 per MTok; one daily report costs well under a cent. Do not reach for a larger model without a measured reason |
| GitHub Actions | Free minutes on public repos, the schedule lives in the repo, and the run log is public evidence the thing actually runs |

### Cost target

| Line | Monthly |
|---|---|
| GitHub Actions — public repo | $0 |
| HF Hub model hosting — public repo | $0 |
| Datasets, RSS feeds, inference | $0 |
| **Total, offline mode** | **$0.00** |
| *Anthropic API, if online mode is enabled — ~22 reports × ~6K in / ~1.5K out, Haiku 4.5* | *~$0.30* |

**In offline mode this project costs nothing.** No key, no prepaid balance, no account beyond GitHub and Hugging Face. Only enable online mode if you decide the narrative paragraph is worth it — and set a spend limit in the Anthropic Console before T4.1 if you do. See `PREREQUISITES.md` §1.

---

## 3. Repo layout and conventions

```
newsmood/
├── README.md                   # what it is → demo → results → install
├── CLAUDE.md                   # persistent rules for Claude Code
├── IMPLEMENTATION_GUIDE.md     # this file
├── HANDOFF.md
├── Makefile
├── pyproject.toml              # entry point: newsmood = "newsmood.cli:app"
├── .github/workflows/daily.yml
├── config/
│   └── default.yaml            # thresholds, feeds, model id, gates
├── src/newsmood/
│   ├── cli.py                  # typer app: ingest | score | report | train | eval
│   ├── config.py               # settings object, env > yaml > default
│   ├── preprocessing.py        # baseline-only cleaning. See the warning below
│   ├── data/
│   │   ├── phrasebank.py       # HF dataset load + splits
│   │   ├── feeds.py            # RSS fetch + normalize
│   │   └── store.py            # SQLite schema + upsert + query
│   ├── models/
│   │   ├── loader.py           # HF Hub download, cache, version pin
│   │   └── classifier.py       # batched predict()
│   ├── baselines/
│   │   ├── vader.py
│   │   └── logreg.py
│   ├── evaluation/
│   │   ├── metrics.py          # ported from the archived repo
│   │   └── gates.py
│   ├── reporting/
│   │   ├── aggregate.py         # mood index arithmetic
│   │   ├── claude.py           # Anthropic client, retry, cost log
│   │   ├── explain.py          # computed lede + Why blocks (T4.2b)
│   │   └── template.py         # markdown assembly
│   └── training/
│       ├── dataset.py          # ported: SentimentDataset
│       └── finetune.py
├── tests/
├── data/sample/headlines_200.jsonl   # committed, for `make demo`
├── reports/                    # committed daily output
└── docs/
    ├── dataset.md              # written by T1.2
    ├── baselines.md            # T1.5
    ├── model_card.md           # T2.6
    ├── drift.md                # T3.4
    └── sources.md              # T3.1
```

### Conventions

- Python 3.11. Dependencies in `pyproject.toml`. Optional extras: `[train]` pulls torch/transformers/datasets so that a user who only wants to *run* the classifier does not install a training stack.
- Config precedence: env var → `config/default.yaml` → code default. Never hardcode a threshold in a module.
- Commit messages: `feat: distilbert-finetuning-complete`, `docs: model-card`. One commit per task, on `main`.
- Every module gets a test. **The quality gates, the metrics, and the mood-index arithmetic must be tested**, because those are the parts that fail silently and still produce a plausible-looking report.
- **No test calls the Anthropic API.** `reporting/claude.py` takes an injectable client; tests pass a fake that returns a canned response. A test suite that costs money is a test suite people stop running.

### Reused code

Ported from the archived `social_media_sentiment_analysis` repo. Copy these in, do not rewrite:

| From | To | Change |
|---|---|---|
| `calculate_comprehensive_metrics()` | `evaluation/metrics.py` | Add macro-F1 alongside weighted |
| `perform_cross_validation()` | `evaluation/metrics.py` | Baselines only |
| `get_device()` | `training/finetune.py` | As-is |
| `EarlyStopping` | `training/finetune.py` | As-is |
| `SentimentDataset` | `training/dataset.py` | `max_length` 128 → 192 |
| AdamW `lr=2e-5, wd=0.01` + linear warmup 10% | `training/finetune.py` | As-is. These are correct |
| VADER ±0.05 compound thresholds | `baselines/vader.py` | As-is |
| `TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.95)` | `baselines/logreg.py` | `max_features` 1000 → 20000 |

**Do not port:** `advanced_text_cleaning()` and its five helpers, the 12-class `sentiments_grouped` map, SMOTE, the LSTM class, the SVM grid search.

### ⚠️ The preprocessing rule

**Raw text goes to the transformer. Cleaned text goes only to TF-IDF.**

The archived repo's cleaning pipeline lemmatizes, strips punctuation, and rewrites negations as `NOT_word`. All three fight DistilBERT's WordPiece tokenizer, which was pretrained on natural text. `preprocessing.py` exists for the TF-IDF baseline and contains only: lowercase, collapse whitespace, strip URLs and ticker symbols. If a future task adds a cleaning step in front of the transformer, that is a regression.

### Sources

| Option | Verdict |
|---|---|
| **RSS via `feedparser`** | **Chosen.** No key, no quota, real-time, no ToS issue |
| NewsAPI.org free tier | Rejected. Dev-use-only in the terms, articles delayed 24 h, content truncated to ~200 chars, 100 req/day. The next tier is $449/month |
| Twitter / X via Tweepy | Rejected. Free-tier read access has been effectively closed since 2023. This was in the original plan and is not viable |
| Reddit via PRAW | Optional stretch in T3.6. Works, but r/stocks comments are a different text distribution from the training corpus, which makes the drift problem worse, not better |

---

## 4. Phase 1 — Package skeleton, data, baselines

Everything local. No API keys needed this week.

### T1.1 — Repo skeleton and first commit
**Depends on:** nothing
**Commit:** `feat: package-skeleton`

`.gitignore` is the **first file in the first commit**, containing at minimum:

```
newsmood_env/
.env
data/*
!data/sample/
/models/
reports/*.json
*.db
__pycache__/
*.egg-info/
.pytest_cache/
*.ipynb_checkpoints
```

`newsmood_env/` has no leading dot, so it is visible in the repo root and **must** be ignored explicitly — a dotless venv directory is the easiest thing in this project to commit by accident.

Then the layout in §3 with empty modules, a `pyproject.toml` with the `newsmood` entry point, and a `cli.py` where every subcommand exists and prints `not implemented`.

`CLAUDE.md` is already in the repo root and is **not** written by this task. Build the tree to match §3 above and the module-boundary rules in `CLAUDE.md`, not from memory.

**Done when:** `pip install -e .` succeeds and `newsmood --help` lists `ingest`, `score`, `report`, `train`, `eval`.

---

### T1.2 — Dataset and the nesting trap
**Depends on:** T1.1
**Commit:** `feat: phrasebank-dataset-setup`

`data/phrasebank.py` loads `takala/financial_phrasebank` from the Hugging Face
Hub. ⚠️ Not via `datasets.load_dataset(...)` — that repo only ships a legacy
loading script, and `datasets>=4.0` dropped script-based loading. Download
`FinancialPhraseBank-v1.0.zip` directly and parse it (`iso-8859-1`, split on
the last `@`), same as the script did. See §9 Deviations, 2026-09-17.

⚠️ **Methodological constraint that propagates through the entire project.** The dataset ships four configs by annotator agreement, and **they are nested supersets**:

```
sentences_allagree (2,264)  ⊂  sentences_75agree (3,453)
                            ⊂  sentences_66agree (4,217)
                            ⊂  sentences_50agree (4,846)
```

Training on one config and evaluating on a smaller one is **training on your test set**. Resolve it this way and do not change it later:

- **Train on `sentences_75agree` only.** It is the largest config where three of four annotators agreed — enough data to fine-tune, clean enough to learn from.
- **Split within that config**: 70/15/15 stratified, `random_state=42`, written to disk so every later task sees identical splits.
- **Never evaluate on another config.** If you want a second number, say "trained and tested on 75agree" and leave it there.

There is no train/val/test split in the source data; you are creating it.

Write `docs/dataset.md` from the actual downloaded files:

- row count per config, confirming the nesting above
- label distribution in `sentences_75agree` — expect roughly 60% neutral, but **record the real number, because it is the majority-class baseline every later accuracy claim is measured against**
- token-length distribution of `sentence` under the DistilBERT tokenizer: mean, p95, max. This sets `max_length`
- 5 example sentences per class, quoted, so the reader can see what the labels mean

Also build `data/sample/headlines_200.jsonl` — 200 held-out sentences, committed, used by `make demo`.

**Done when:** `docs/dataset.md` has real numbers, and `python -m newsmood.data.phrasebank --summary` prints the three split sizes and the majority-class rate.

---

### T1.3 — VADER baseline
**Depends on:** T1.2
**Commit:** `feat: vader-baseline`

Port the ±0.05 compound thresholds. Run on the test split.

Expect this to do **badly** — somewhere near or below the majority-class rate. VADER's lexicon has no entry for "impairment", scores "liability" as neutral, and reads "aggressive growth" as negative. That result is the point: it motivates everything after it. Record the number even though it is bad, and record three specific sentences VADER gets wrong for `docs/baselines.md`.

**Done when:** `newsmood eval --model vader` prints accuracy, macro-F1, and a 3×3 confusion matrix.

---

### T1.4 — TF-IDF + Logistic Regression baseline
**Depends on:** T1.3
**Commit:** `feat: logistic-regression-baseline`

`preprocessing.py` (the minimal version, §3) → `TfidfVectorizer` → `LogisticRegression(class_weight="balanced", max_iter=1000)`.

**Fit the vectorizer on the training split only.** Fitting on all data before splitting leaks test vocabulary and IDF statistics into training, which inflates the number and is the most common silent error in text-classification portfolios.

No SMOTE. `class_weight="balanced"` handles the imbalance without inventing sentences that were never written.

**Done when:** `pytest tests/test_logreg.py` passes, including a test that asserts the vectorizer was not fit on test data.

---

### T1.5 — Metrics module and the baseline floor
**Depends on:** T1.4
**Commit:** `feat: baseline-evaluation`

Port `calculate_comprehensive_metrics()`. Add three reference rows that every later table must carry:

| Row | What it is |
|---|---|
| **Majority class** | Predict "neutral" always. This is the floor |
| **Stratified random** | Sample from the training label distribution |
| **Human ceiling** | The 25% of annotators who disagreed on 75agree sentences imply an upper bound. Note it qualitatively; do not fabricate a number |

**A number without a floor is not a result.** "DistilBERT hits 82%" means nothing until the reader knows that always-guessing-neutral scores ~60%.

**Done when:** `newsmood eval --all` writes `docs/baselines.md` with one row per method × {accuracy, macro-F1, weighted-F1, per-class recall}.

---

### T1.6 — Baseline analysis writeup
**Commit:** `docs: baseline-analysis`
Write the analysis paragraph in `docs/baselines.md`: why VADER fails on financial text specifically, and what the gap between LogReg and the majority floor tells you about how much signal is in surface vocabulary alone.

---

## 5. Phase 2 — Fine-tuning and shipping the model

### T2.1 — Training scaffolding
**Depends on:** T1.5
**Commit:** `feat: distilbert-framework-setup`

`training/dataset.py` — port `SentimentDataset`, `max_length` from T1.2's
measured p95: **58** (mean 30.4, max 150 — see `docs/dataset.md`). Placeholder
of 192 superseded, §9 Deviations 2026-09-17.

`training/finetune.py` — port `get_device()` and `EarlyStopping`. Use:

```python
AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    id2label={0: "negative", 1: "neutral", 2: "positive"},
    label2id={"negative": 0, "neutral": 1, "positive": 2},
)
```

Not the archived repo's custom `nn.Module` with a hand-rolled `[CLS]` head. The custom class works, but `AutoModelForSequenceClassification` gives you `save_pretrained` / `from_pretrained`, which means the artifact is a directory anyone can load in two lines without importing your class at the right version. That matters more than the twelve lines it saves.

Weighted `CrossEntropyLoss` with weights from the training split.

**Done when:** `newsmood train --dry-run --limit 32` completes one forward and one backward pass on CPU.

---

### T2.2 — Fine-tune
**Depends on:** T2.1
**Commit:** `feat: distilbert-finetuning-complete`

3 epochs, batch 16, `lr=2e-5`, `weight_decay=0.01`, linear warmup over 10% of steps. Log train loss, val loss, val accuracy per epoch to `models/local/history.json`.

⚠️ **Needs a long session — scheduled for Sunday Sep 27.** On a laptop CPU, 3 epochs over ~2,400 training sentences runs 20–40 minutes. It fits a 1–2 hour Sunday alongside T2.3; it does not fit a 30-minute weekday slot. Published DistilBERT results on this dataset land around 82%; the 80% target is realistic but not automatic.

Start the run at the top of the session and read `docs/dataset.md` while it trains — you will be asked about this data in an interview, and the wait is free reading time.

If val accuracy sits at the majority-class rate after epoch 1, stop and check the label mapping before changing hyperparameters. That failure mode — a model that has learned to always say "neutral" — looks like a training problem and is almost always a data problem.

**Done when:** `models/local/` contains a `save_pretrained` directory and val accuracy exceeds the majority-class floor by at least 10 points.

---

### T2.3 — Evaluate against everything
**Depends on:** T2.2
**Commit:** `feat: model-comparison`

Same test split, same metrics module. Add one extra row: **`ProsusAI/finbert` zero-shot**, as a reference ceiling. It was pretrained on financial text, so beating it is not expected and not the goal — the goal is showing how close a generic model fine-tuned on 2,400 sentences gets to a domain-pretrained one.

Include the confusion matrix in `docs/baselines.md` and write a sentence about which class the model confuses most. On this dataset it is usually negative→neutral, because a headline reporting a loss in flat language reads neutral to a model that has only seen 300 negative examples.

**Done when:** `newsmood eval --all` shows five rows: majority, VADER, LogReg, DistilBERT, FinBERT-zeroshot.

---

### T2.4 — Publish to the Hub
**Depends on:** T2.3
**Commit:** `feat: model-hub-publish`

`huggingface-cli upload` the `save_pretrained` directory to `<your-username>/newsmood-distilbert-financial`. Public.

`models/loader.py` downloads on first use, caches under `~/.cache/newsmood/`, and **pins a revision SHA** in `config/default.yaml`. An unpinned `main` means the model can change under a user without their version changing, which makes every reported number unreproducible.

Print a one-line notice on first download: size, destination, and that it happens once.

**Done when:** on a machine with the cache deleted, `newsmood score --text "Profit fell sharply"` downloads the model and prints `negative` with a confidence.

---

### T2.5 — `newsmood score`
**Depends on:** T2.4
**Commit:** `feat: batched-inference`

Batched CPU inference, batch size 32, `torch.inference_mode()`. Accepts `--text`, `--file`, or reads unscored rows from SQLite.

Measure and record in `docs/model_card.md`: throughput in headlines/second, p50 and p95 per-headline latency batched, and single-headline cold latency.

The original plan's "<500 ms per document" target will be met by roughly an order of magnitude, which makes it a weak claim. **Report throughput instead** — "1,000 headlines in ~20 seconds on a 2020 laptop CPU, no GPU" is a real number that a reader can size against their own hardware.

**Done when:** `newsmood score --file data/sample/headlines_200.jsonl --benchmark` prints throughput and latency percentiles.

---

### T2.6 — Model card
**Commit:** `docs: model-card`
Write `docs/model_card.md`: intended use, training data and its provenance, the split policy from T1.2, metrics with the baseline floor, known limitations, and **an explicit "not financial advice" line**. A model card is table stakes for anyone who has worked with models professionally, and almost no student project has one.

---

## 6. Phase 3 — Ingest and the honesty tasks

### T3.1 — RSS ingest
**Depends on:** T2.5
**Commit:** `feat: rss-ingest`

`data/feeds.py` using `feedparser`. Feed list lives in `config/default.yaml`, not in code. Candidates to test:

```yaml
feeds:
  - name: yahoo-finance
    url: https://finance.yahoo.com/news/rssindex
  - name: cnbc-finance
    url: https://www.cnbc.com/id/10000664/device/rss/rss.html
  - name: marketwatch-top
    url: https://feeds.content.dowjones.io/public/rss/mw_topstories
```

**Verify each one returns entries before committing it.** Feed URLs rot. Write `docs/sources.md` recording, per feed: whether it responded, entries per fetch, fields present, and typical headline length. Drop the dead ones and say so.

Normalize to: `id` (sha256 of url), `source`, `title`, `summary`, `url`, `published_at` (UTC), `fetched_at`.

Set a descriptive `User-Agent`. Fetch politely, sequentially, with a timeout.

**Done when:** `newsmood ingest --dry-run` prints per-feed entry counts and `docs/sources.md` records which feeds are live.

---

### T3.2 — SQLite store
**Depends on:** T3.1
**Commit:** `feat: sqlite-store`

One table `headlines`, PK on the content hash, so re-running ingest is idempotent. Columns for `label`, `confidence`, `scored_at`, nullable until scoring runs.

Dedupe on **normalized title**, not URL — the same wire story appears at five outlets with five URLs, and counting it five times skews the daily index toward whatever the wires ran.

**Done when:** `pytest tests/test_store.py` passes, including a test that ingesting the same feed twice leaves the row count unchanged.

---

### T3.3 — Mood index
**Depends on:** T3.2
**Commit:** `feat: daily-mood-index`

Aggregate the day's scored headlines into one number in `[-100, +100]`:

```
mood = 100 × Σ(wᵢ · sᵢ) / Σ(wᵢ)
  sᵢ = +1 positive, 0 neutral, -1 negative
  wᵢ = confidence, if ≥ threshold; else 0
```

Confidence threshold in config, default 0.60. Report the count of headlines dropped below threshold — a day where half the headlines were low-confidence is a day the index should not be trusted, and hiding that is how a dashboard lies.

Store the daily value so reports can show a trend.

**Done when:** `pytest tests/test_index.py` passes, with cases for all-positive, all-neutral, empty day, and all-below-threshold.

---

### T3.4 — Out-of-domain reality check
**Depends on:** T3.3
**Commit:** `docs: domain-drift-analysis`

**This is the most valuable task in the project. Do not skip it if the schedule slips.**

The model is trained on curated analyst sentences from a Finnish corpus assembled in 2014. It will run on 2026 wire headlines. These are not the same distribution: headlines are shorter, use present tense and dropped articles, and carry more proper nouns.

By hand, label 100 real ingested headlines. Do it in one sitting so your own labeling is consistent. Then measure the model's accuracy against your labels and compare it to the T2.3 test accuracy.

**Expect a drop.** Ten to twenty points would be unsurprising. Write `docs/drift.md` with the two numbers side by side, the confusion matrix on real headlines, and five examples the model gets wrong with a sentence on why.

Publishing the number that makes your model look worse is what separates this from the portfolio projects that report one test accuracy and stop. It is also the thing an interviewer can ask three follow-up questions about.

**Done when:** `docs/drift.md` exists with both accuracies and the 100 labels committed as `data/sample/hand_labeled_100.jsonl`.

---

### T3.5 — Quality gates
**Depends on:** T3.4
**Commit:** `feat: quality-gates`

`evaluation/gates.py`. Each gate returns pass/fail plus the observed value. **A failing gate aborts before the Claude call and before the report is written.** A pipeline that publishes a confident report from three headlines is worse than one that stops.

```yaml
gates:
  min_headlines: 15
  max_duplicate_rate: 0.40
  min_sources_responding: 2
  max_low_confidence_rate: 0.50
  max_single_source_share: 0.70
  max_feed_staleness_hours: 36
```

Write `quality_report.json` every run, pass or fail.

**Done when:** `pytest tests/test_gates.py` passes, with a test per gate that deliberately feeds it failing data. **These tests matter more than any others in the repo** — an untested gate waves the bad run through.

---

### T3.6 — Reddit ingest *(optional, drop first if behind)*
**Commit:** `feat: reddit-ingest`

PRAW against r/stocks and r/investing. Only worth doing if T3.4 came in strong. Tag rows with `source_type` so the mood index can be computed with and without social text — mixing wire headlines and Reddit comments into one number without that flag makes the number uninterpretable.

---

### T3.7 — Pipeline runbook
**Commit:** `docs: pipeline-runbook`
How to run each stage manually, what each gate failure means, how to re-score after a model update, how to backfill a missed day.

---

## 7. Phase 4 — Reports, automation, release

**Phase 4 is offline-first.** T4.2, T4.3, T4.5 and T4.6 need no API key. T4.1 and the online path are optional and come last.

**Before T4.1 only:** set a spend limit in the Anthropic Console. See `PREREQUISITES.md` §1.

### T4.1 — Claude client
**Depends on:** T4.3
**Commit:** `feat: claude-api-integration`

**Optional.** Everything ships without this. Do it only once the offline report works end to end.

This task creates `.env.example` — a committed file holding the variable *name* with an empty value, so a reader knows the variable exists. The real key lives in `.env`, which is gitignored from T1.1 onward and never committed. Until this task runs, the project has no environment variables and no `.env.example`.

`reporting/claude.py`. Model `claude-haiku-4-5-20251001`, pinned in config.

- Client is **injectable** so tests pass a fake. No test hits the network.
- Retry with exponential backoff on 429 and 5xx.
- Log tokens in/out per call to `reports/costs.jsonl`. You cannot manage a bill you do not measure, and this file is also the evidence for the cost claim in the README.
- `--dry-run` prints the assembled prompt and exits without calling. Use it while iterating on wording; every prompt tweak otherwise costs a call.

**The model summarizes; it does not classify.** Sentiment labels come from DistilBERT and are passed in as given. If Claude is allowed to re-judge sentiment, the fine-tuned model becomes decorative and the project's central claim evaporates.

**Done when:** `pytest tests/test_claude.py` passes with a mocked client, and `newsmood report --dry-run` prints a prompt.

---

### T4.2 — Report template
**Depends on:** T4.1
**Commit:** `feat: report-template`

Deterministic markdown assembled in `reporting/template.py`, with exactly one Claude-generated section inside it:

```markdown
# Newsmood — {date}

**Mood index: {score:+.1f}**  ({label})   {n} headlines from {k} sources

| Sentiment | Count | Share |
...

## What happened
{claude narrative — 150-200 words}

## Headlines
| Sentiment | Conf | Headline | Source |
...

---
Model: {model_id}@{revision} · Gates: {passed}/{total} · Not financial advice.
```

Counts, tables, and the index are computed in Python, never asked of the model. Ask a language model to add up a column and it will sometimes be wrong, silently, in a document that looks authoritative.

The footer's model id and revision are what make a report from six weeks ago interpretable.

---

### T4.3 — End-to-end report
**Depends on:** T4.2, T2.5, T3.3
**Commit:** `feat: automated-report-generation`

`newsmood report [--date YYYY-MM-DD]` runs ingest → score → gates → narrative → write, and is safe to re-run for the same date.

**Build `--offline` first and make it the default for now.** In offline mode the narrative section is a short placeholder line and every other section is fully computed. T4.1 is not a dependency of this task; wire the Claude call in later behind the flag.

**Done when:** `newsmood report --offline` produces `reports/{today}.md` from live RSS, with a real mood index, real counts, and the full scored headline table — no API key present anywhere in the environment.

---

### T4.2b — The "Why" section, computed
**Depends on:** T4.3
**Commit:** `feat: computed-explanation-section`

Turns the report from a table into something a reader can act on, **with no API call**. Everything here is arithmetic over artifacts that already exist.

Five blocks in `reporting/explain.py`. **Block 0 is written last**, because it composes from the other four.

**0. The lede — one sentence, assembled from a template.**

A single human-readable line at the top of the report. No model, no API, no download: it is string assembly over values blocks 1–4 already produced, so it is free, instant, and unit-testable.

```
Financial news leaned moderately negative today (index -18.4, down 12.1
from yesterday), driven mainly by 9 headlines on foundry delays, with
regional bank earnings the main counterweight.
```

Four slots, each independently omitted when its data does not support it:

| Slot | Source | Omitted when |
|---|---|---|
| Direction + strength | index band (see below) | never — always present |
| Change | delta vs. previous session | \|delta\| < 5, or no prior session |
| Driver | largest cluster (block 3) + its distinctive terms (block 2) | cluster below `min_cluster_size` or below 70% single-sentiment |
| Counterweight | top mover on the opposite side (block 1) | no opposite-side headline above threshold |

Bands in `config/default.yaml`, so the wording is tunable without touching code:

```yaml
lede_bands:
  flat: 5          # |index| < 5
  mild: 15
  moderate: 35     # above this, "strongly"
```

With every optional slot dropped, the sentence still stands on its own:

```
Financial news was roughly flat today (index +1.2) across 39 headlines,
with no dominant theme.
```

⚠️ **The template must never assert a driver it cannot evidence.** The "driven mainly by" clause appears only when a real cluster clears both thresholds. A sentence that names a theme on four loosely related headlines reads exactly like a sentence that names a real one, and the reader has no way to tell them apart. When in doubt the sentence gets shorter, never vaguer.

Keep a small phrase bank (2–3 variants per slot, chosen by hash of the date) so consecutive days do not read identically. This is cosmetic, not semantic — variants must be interchangeable in meaning.

**This is what makes the report readable without the API.** The Claude narrative in T4.1, if you build it, replaces this one sentence with a paragraph. It does not replace the evidence blocks below it.

---

**1. Top movers, both directions.** Each headline's term in the weighted sum from T3.3, as a percentage of the day's total movement. This is not a new computation — it is the index, itemized.

**Show the top 3 negative and the top 3 positive as separate blocks, always, even on a lopsided day.** Ranking by absolute contribution alone would fill all five slots with one side on most days, and a reader scanning before the open needs both: what is dragging, and what is holding up. On a day with no positive headlines at all, print "no positive headlines above the confidence threshold" rather than omitting the block — an absent section and an empty one mean different things.

⚠️ **Label this by what it is: most confidently classified, not most important.** The model scores tone, not market significance. A 0.97-confidence headline about a small-cap is not more consequential than a 0.71-confidence headline about the Fed, and the ranking cannot tell them apart. Put that caveat in the section header, not in a footnote — it is the most likely way a reader will misuse this report.

**2. Distinctive terms.** Reuse the **T1.4 TF-IDF vectorizer**. Score today's negative headlines against a trailing 30-day background and surface the terms that stand out. The baseline you built in Week 1 earns a second job here.

**3. Clusters.** Cosine similarity on the same TF-IDF vectors, agglomerative, distance threshold in config. For each cluster: size, sentiment split, and share of the day's tilt.

**4. Source skew.** Share of negative headlines from the single most represented outlet. A day where one wire dominates is a day the index reflects one newsroom.

⚠️ **Say when there is nothing to say.** At ~47 headlines a day, distinctive-term extraction is noisy and clustering is coarse. Each block needs a floor — minimum headlines, minimum cluster size, minimum term lift — and prints an explicit "no clear theme today" when unmet. **Manufacturing a theme from four headlines is worse than printing nothing**, because the reader cannot tell the difference and will believe it.

Until the trailing window has 30 days of history, block 2 degrades to raw term frequency and says so in the output.

**Done when:** `pytest tests/test_explain.py` passes, including a case with 5 headlines where every block correctly declines to report and the lede degrades to its shortest form, and `newsmood report --offline` shows a populated lede and Why section on real data.

---

### T4.4 — GitHub Actions
**Depends on:** T4.3
**Commit:** `feat: daily-automation`

Run `newsmood report --offline` unless the online path is done. `.github/workflows/daily.yml`, `cron: "0 22 * * 1-5"` (22:00 UTC, after the US close). Also `workflow_dispatch` so you can trigger it by hand.

- `ANTHROPIC_API_KEY` from repo secrets, **only if online mode is enabled**. Never echoed.
- Cache `~/.cache/huggingface` — otherwise every run re-downloads 265 MB.
- Commit the report back with a bot identity, `[skip ci]` in the message.
- On failure, write the gate report to the job summary so the run page shows *why*.

⚠️ **Needs a long session — scheduled for Sunday Oct 18.** Budget 45–60 minutes. First-run YAML iteration always overruns, and each attempt costs a push and a wait you cannot shorten.

**Done when:** a manual `workflow_dispatch` produces a committed report, and a run with `min_headlines` temporarily set to 9999 fails visibly with the gate named.

---

### T4.5 — `make demo` and README
**Depends on:** T4.4
**Commit:** `docs: final-documentation`

`make demo` must work on a clean clone **with no API key**:

```
pip install -e . && \
newsmood score --file data/sample/headlines_200.jsonl && \
newsmood report --offline --input data/sample/headlines_200.jsonl
```

`--offline` skips the Claude call and emits the report with a placeholder narrative. Everything else is real.

**This is the single highest-leverage thing in the repo.** Most portfolio projects cannot be run by the person evaluating them. Yours can, in under two minutes, with no signup.

README order: one sentence → sample report screenshot → `make demo` → results table with the baseline floor → the drift number from T3.4 → install → how it works → **using the data** → engineering notes. Installation goes near the end; nobody installs before deciding they care.

**Using the data** — a short section, three or four lines, no new command:

> Scored headlines and the daily index live in a SQLite file at `data/newsmood.db`.
>
> - `headlines` — `id, source, title, url, published_at, label, confidence, scored_at`
> - `daily_index` — `date, mood_index, n_headlines, n_positive, n_neutral, n_negative, n_low_conf, n_sources`
>
> ```python
> pd.read_sql("SELECT * FROM daily_index", sqlite3.connect("data/newsmood.db"))
> ```

This serves anyone who wants the series for their own work without adding an export command to build, document and maintain. If you later find yourself wanting a CSV after real use, add it then — you will know the shape you want.

Link the archived `social_media_sentiment_analysis` repo as the predecessor, with one line on what changed and why.

---

### T4.6 — Release
**Commit:** `docs: release`
Tag `v0.1.0`. Optionally publish to PyPI (`newsmood` may be taken — check first; `newsmood-cli` as fallback, keeping `newsmood` as the import and command). A GitHub release with the demo output attached is sufficient if PyPI is a hassle.

---

## 8. Interview material this produces

Keep these in the README under "Engineering notes."

- **A measured floor.** Every accuracy in the repo is reported against the majority-class rate. Most candidates report 82% on a 60%-neutral dataset and do not mention the 60%.
- **Nested-config leakage avoided.** The four phrasebank configs are supersets; training on 50agree and testing on allagree is training on your test set. Documented in T1.2 and enforced by the split policy.
- **Out-of-domain honesty.** T3.4 hand-labels real headlines and publishes the drop. This is the answer to "how do you know it works in production," and it is a question most student projects cannot answer at all.
- **Right tool per job.** The fine-tuned model classifies; the LLM writes prose; Python does the arithmetic. Each does the thing it is reliable at.
- **Gates with teeth, and tests for every gate.** The pipeline refuses to publish a confident report from thin data.
- **Cost measured, not estimated.** `reports/costs.jsonl` is the receipt.
- **Rightsizing, again.** No cloud, no cluster, no GPU. A few MB of text a month does not need infrastructure, and knowing when *not* to reach for it is a screened-for skill.

---

## 9. Deviations from this spec

Append whenever reality differs. Date, task ID, what changed, why.

| Date | Task | Change | Reason |
|---|---|---|---|
| 2026-09-16 | T1.1 | `.gitignore` line 3 changed from `data/` to `data/*` | Git cannot re-include a path (`!data/sample/`) inside a directory that is itself excluded — it never descends into `data/` to check per-file rules. The literal spec pattern silently ignored `data/sample/` too, which would have blocked committing `data/sample/headlines_200.jsonl` in T1.2. |
| 2026-09-16 | T1.1 | `.gitignore` `models/` line changed to `/models/` | Unanchored `models/` matches any directory named `models` at any depth, including `src/newsmood/models/` (real source code per §3). It silently ignored the whole models sub-package. Anchoring to repo root limits it to the checkpoint directory it was meant to exclude. |
| 2026-09-17 | T1.2 | `data/phrasebank.py` loads PhraseBank by downloading `FinancialPhraseBank-v1.0.zip` directly and parsing it, not via `datasets.load_dataset("takala/financial_phrasebank", ...)` | The HF repo only ships a legacy loading script (no Parquet conversion); `datasets>=4.0` (installed: 5.0.1) dropped script-based loading entirely (`RuntimeError: Dataset scripts are no longer supported`). The replacement parses the same zip with the script's own logic (`iso-8859-1`, split on last `@`) — same data, same four configs, not a new source. |
| 2026-09-17 | T1.2 | `max_length` guidance for T2.1 revised from the 192 placeholder to the measured p95 (58) | Real token-length distribution under `distilbert-base-uncased`: mean 30.4, p95 58, max 150 (see `docs/dataset.md`). 192 would pad most batches to 3x+ the length actually needed. |
| 2026-09-17 | T1.2 | Added `[project.optional-dependencies] dev = ["pytest"]` to `pyproject.toml` | T1.1's skeleton never declared `pytest` anywhere, but every task's Done-when bar from here on runs `pytest`. Left undeclared, `pytest tests/` only worked by accident of whatever happened to be on the machine already. |
| 2026-09-18 | T1.3 | `config.py`'s `settings_customise_sources` reordered to `init_settings, env_settings, YamlConfigSource, dotenv_settings, file_secret_settings` — explicit kwargs > env > yaml > code default | pydantic-settings ranks sources by position (earlier wins); the T1.1 order (`env, yaml, init, ...`) meant any field `config/default.yaml` defines silently overrode explicit constructor kwargs, e.g. `Settings(dataset=DatasetSettings(splits_dir="/tmp/x", ...))` returned `data/splits` regardless. T1.2's tests never surfaced this because they always passed explicit `Path` args around the settings object instead of reading a settings-derived path back out. `baselines/vader.py`'s `predict_test_split(settings)` is the first function to resolve a path from `settings` the way the CLI does, and it silently loaded the real 518-row test split instead of a tmp fixture. First pass only moved `init_settings` above yaml and left `env_settings` ahead of it, still wrong for the same reason one tier up (a stray `NEWSMOOD_*` env var could override an explicit test kwarg) — caught on review and corrected same day. Now tested directly in `tests/test_config.py` (pairwise per tier, plus a `field-default` tier probed via a throwaway `BaseSettings` subclass, since no real settings field currently has a Python-level default). Full suite re-run after the change, including `tests/test_phrasebank.py` (reads `splits_dir` and `seed` from settings) — all tests pass, `newsmood eval --model vader` and `python -m newsmood.data.phrasebank --summary` produce identical output to before the reorder. |
| 2026-09-21 | T1.4 | `pyproject.toml` gained `scikit-learn>=1.9,<2` as a core dependency, not behind `[train]` | Never declared anywhere despite `baselines/logreg.py` importing `TfidfVectorizer`/`LogisticRegression` directly. It can't sit behind `[train]` with torch/transformers/datasets: `reporting/explain.py` loads the persisted vectorizer at report time in T4.2b (distinctive terms, clustering), so a plain `pip install newsmood` running `newsmood report` needs it too, with no fine-tuning involved. Version bound tied to the installed/tested version (1.9.1) — the joblib artifacts in `models/baselines/` are pickles, which can fail to load across scikit-learn versions; `meta.json` alongside them also records the exact version they were fit under. |
