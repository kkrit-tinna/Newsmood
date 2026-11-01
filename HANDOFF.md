# Newsmood — session handoff

*Read this first, every session, before touching code. Then `CLAUDE.md` for the standing rules, and `IMPLEMENTATION_GUIDE.md` for the one task you're doing.*

---

## What this is

`newsmood` — a Python CLI that pulls financial news headlines from public RSS feeds, classifies each one positive / neutral / negative with a DistilBERT model fine-tuned on the Financial PhraseBank, aggregates them into a daily mood index, and writes a markdown report per weekday. The report's lede sentence and explanation blocks are computed in Python. A Claude-written narrative paragraph is optional and off by default.

Successor to `social_media_sentiment_analysis`, which is archived. That project stalled on a synthetic Kaggle dataset with 191 unusable sentiment labels; the models topped out at 56%. This one uses a real corpus with three clean classes and ships as an installable tool rather than a notebook.

**Timeline:** Sep 16 – Oct 30, 2026 (extended from Oct 18 on Oct 8 for school and the Oct 15 career fair, then from Oct 22 on Oct 23 after Oct 15–25 went to ad hoc work). **Offline-first** — first working report reached Fri Oct 2.

---

## Working context

I'm comfortable with pandas, scikit-learn, and notebook-scale PyTorch. **I have not shipped a pip-installable package before, and I have not fine-tuned a transformer to a working result.** Explain packaging, entry points, extras, and HF Hub publishing as they come up.

---

## Decisions already made — do not relitigate these unless I ask

**1. A CLI package, not a web service. No cloud, no Docker, no Terraform.**
Storage is SQLite. Scheduling is GitHub Actions.

**2. The model ships from the Hugging Face Hub, not inside the wheel.**
A fine-tuned DistilBERT is ~265 MB against PyPI's 100 MB file limit. Lazy download to `~/.cache/newsmood/` on first run, revision pinned in config.

**3. RSS feeds, not a news API and not Twitter.**
NewsAPI's free tier is development-use-only with a 24-hour delay and truncated content. Twitter free-tier read access has been closed since 2023. RSS has no key, no quota, and no terms problem.

**4. One PhraseBank config only — `sentences_75agree`.**
The four agreement configs are **nested supersets** (verified by set containment in T1.2). Training on one and evaluating on a smaller one is testing on training data. Split within the one config, 70/15/15 stratified, seed 42. **This constraint propagates through the whole project.**

**5. Raw text to the transformer and to VADER. Cleaned text only for TF-IDF.**
Lemmatizing, stripping punctuation and rewriting negations fight WordPiece tokenization. VADER keys on capitalization and punctuation, so cleaning would cripple the baseline and flatter the model.

**6. DistilBERT fine-tuned, not FinBERT.**
FinBERT is already trained on financial sentiment, so using it means there is no fine-tuning story. FinBERT appears once,FinBERT appears once, in T2.3, as a reference only: it was trained on PhraseBank, so its test score is neither zero-shot nor a ceiling.

**7. Claude writes prose, if enabled. It never classifies.**
Labels come from the fine-tuned model. Counts and the index are computed in Python. If an LLM re-judges sentiment or adds up a column, the project's central claim disappears.

**8. No SMOTE.** Class weights instead. Interpolating between TF-IDF vectors invents sentences nobody wrote.

**9. Offline-first. The Claude API is optional and comes last.**
`newsmood report --offline` produces the whole report, including a one-sentence template lede and the computed Why section. No API key, no paid service — the project costs $0.00 in this mode. T4.1 would upgrade the lede to a paragraph and nothing else.

**10. Data sources are settled: Financial PhraseBank for training, RSS for daily input. Nothing else.**
Don't reopen this; propose a source only if one of these two breaks.

---

## Target architecture

```
GitHub Actions (weekdays 22:00 UTC)
  └─> newsmood ingest   RSS → normalize → dedupe → SQLite
  └─> newsmood score    DistilBERT from HF Hub (pinned, cached) → label + confidence
  └─> newsmood report   aggregate → gates → computed lede + Why blocks → markdown
                        (optional: Claude narrative, online mode only)
  └─> commit reports/YYYY-MM-DD.md
```

**Cost: $0.00/month in offline mode.**

---

## Where I am right now

Every planned task is done, v0.1.0 released on Oct 31.

Unassigned cleanup:
- `perform_cross_validation()` is unported: assign it or drop it.
- T1.3 lexicon facts are wrong ("liability" −0.8; "aggressive growth" +0.25):
  guide edit + §9.
- `docs/baselines.md` VADER wording (~line 202) assumes one mechanism; the
  line-155 CLAIM splits the negative errors 23/15.
- Optional: one-line RSS summary per article (needs `&amp;` unescaping).
- Minor: `compute_mood` divides by zero (unreachable); `scored_at` uses the real clock.
- `scripts/t34_label.py`'s default path points at the deleted
  `data/sample/t34_blind.jsonl`; harmless, since `--file` is always passed.

**Deviations**: full rows in `IMPLEMENTATION_GUIDE.md` §9
- Setup: `.gitignore` patterns; PhraseBank from a SHA-pinned zip; config
  precedence explicit > env > yaml > default; torch, transformers, datasets,
  tqdm, scikit-learn and huggingface_hub are core; `[train]` is empty
- Training: `max_length` 96 + dynamic padding; EarlyStopping deepcopy fix;
  `get_device()` cuda > mps > cpu; FinBERT relabelled
- Hub and scoring: `hf upload`; cache `~/.cache/newsmood`; `--device`; `pred_label`
- Index (T3.3): ET day; no 0.60 cutoff; no minimum, stats instead; hybrid
  recompute; terms stored
- Report (T4.2/T4.3): redesigned layout; `pipeline.py`; `report.output_dir`;
  late-headline exclusion
- T4.2b: blocks 2 and 4 and the counterweight dropped; `window` vectorizer;
  previous-session rule; driver by share of tilt; clusters at 0.6
- T3.5: gates split into 2 blocks + 5 disclosures; `max_duplicate_rate`
  dropped; staleness compares ≥; Yahoo → Investing.com; T4.4 Done-when rewritten
- Schedule: release moved to Oct 22; T3.6 dropped; T4.1 deferred past release
- Oct 23: release → Oct 30, no sessions Oct 15–25 (no data gap); DB on orphan
  `data` branch (T4.4); `make demo` via env-var paths (T4.5); T3.7 → README;
  T3.4 FinBERT add-on dropped

---

## Reference — built components and invariants

*Stable. Add a line when a task builds something; don't rewrite each session.*

- **Timestamps**: `YYYY-MM-DDTHH:MM:SS.ffffff+00:00`, 32 chars, UTC, via
  `store._ts` only. Day ranges are string comparisons and depend on it.
- **T1.5**: `eval --all` writes the summary table between
  `<!-- eval:summary:start/end -->` in `docs/baselines.md`.
- **T1.6**: interpretive sentences in `docs/baselines.md` §Analysis are
  wrapped in `<!-- CLAIM: ... -->` (`grep CLAIM:`).
- **T2.4**: `loader.ensure_model()` checks the cache first, downloads only on a
  miss, pinned SHA, `cache_dir=~/.cache/newsmood`, `token=False`. Never reads
  `models/local/`. `config.py` rejects any revision that isn't a 40-char SHA.
- **T2.5**: bare `score` writes only `label IS NULL` rows; `--text`/`--file`
  are read-only; `--file` adds `pred_label` + `confidence`. `--benchmark`
  defaults to CPU. Code in `models/scoring.py`, `models/benchmark.py`.
- **T2.5**: Trained on MPS: reproducible in substance, bit-exact only with `--device cpu`.
  Actions scores on CPU, local runs on MPS.
- **T3.2**: `store.py` owns the schema. Dedupe `UNIQUE(title_norm)`, first seen
  wins; `published_at` only moves earlier; label/confidence/scored_at all set
  or all NULL.
- **T3.3**: ET day; every row counts, weighted by confidence; stats stored
  per day; today and yesterday recomputed, older days frozen; empty days not stored.
- **T4.2/T4.3**: `build_report_data()` (all arithmetic) → `render()`
  (formatting only) → `run_report()` writes atomically. Late headlines for a
  frozen day are excluded and counted. Re-runs are byte-identical.
- **T4.2b**: `reporting/explain.py`. Lede = direction + change (previous session:
  D−1…D−5 with n ≥ 5; missing days computed) + driver (largest share of tilt in
  the mood's direction, size ≥ 3, purity ≥ 0.70; clause omitted below
  `min_headlines`). Contribution and share use Σ|conf·s|. `window` TF-IDF fit on
  D plus 30 days, cosine distance, complete linkage at 0.6. Phrase variants by
  SHA-256 of the date.
- **T3.5**: `evaluation/gates.py`, pure. Blocks (`pipeline_fresh`, `all_scored`)
  run after scoring and before `compute_days`; a block exits 1 and writes no
  report. Disclosures (`thin_day`, `feeds_down`, `single_source`,
  `low_confidence`, `stale_feed`) fill one "Data notes" line. Staleness is
  age ≥ threshold. `reports/quality/{date}.json` every run that reaches the gates, pass or fail.
---

## How I want you to work with me

- **`IMPLEMENTATION_GUIDE.md` is the source of truth. One `T#.#` task per session.** Don't skip ahead of a task's dependencies.
- **Start each session by telling me which task you're doing and whether it fits in the time I have.** If it doesn't, propose where to split it.
- If reality contradicts the spec — a row count is off, an RSS feed is dead, a column is missing — **update the guide and log it under §9 Deviations** in the same commit. Don't work around it silently.
- **Explain the packaging and ML-ops parts as they come up.** Entry points, extras, `save_pretrained`, Hub revisions, Actions caching. I'd rather understand the thing than have it work.
- **Push back if I'm over-engineering.** If I ask for a feature that isn't in the guide, ask which task it replaces.
- **Never write a test that calls the Anthropic API.** Inject a fake client.
- **Flag anything that spends money before running it**, including a training run that will heat my laptop for 40 minutes.
- When a metric comes out worse than hoped, **write down the real number**.
- **Keep §Where I am a snapshot, not a log.** Rewrite it each session: status, the numbers table, carry-forward items, deviations as one-liners, next task. Detail belongs in commit messages, §9, and `docs/`.