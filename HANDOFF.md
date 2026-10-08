# Newsmood — session handoff

*Read this first, every session, before touching code. Then `CLAUDE.md` for the standing rules, and `IMPLEMENTATION_GUIDE.md` for the one task you're doing.*

---

## What this is

`newsmood` — a Python CLI that pulls financial news headlines from public RSS feeds, classifies each one positive / neutral / negative with a DistilBERT model fine-tuned on the Financial PhraseBank, aggregates them into a daily mood index, and writes a markdown report per weekday. The report's lede sentence and explanation blocks are computed in Python. A Claude-written narrative paragraph is optional and off by default.

Successor to `social_media_sentiment_analysis`, which is archived. That project stalled on a synthetic Kaggle dataset with 191 unusable sentiment labels; the models topped out at 56%. This one uses a real corpus with three clean classes and ships as an installable tool rather than a notebook.

**Timeline:** Sep 16 – Oct 18, 2026. **Offline-first** — first working report targeted Fri Oct 2.

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
FinBERT is already trained on financial sentiment, so using it means there is no fine-tuning story. FinBERT appears once, as a zero-shot reference ceiling in T2.3.

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

*Last updated: Wednesday, Oct 7, 2026*

**Status:** T1.1–T1.6, T2.1–T2.5, T3.1–T3.3, T4.2, T4.3 and T4.2b complete. 322 tests.
Reports now carry a computed lede, contribution % on recommended articles, and a
cluster block that declines honestly at current volume.

**Next task:** T3.5 quality gates, Thu Oct 8.

**Schedule (remaining)**

| Date | Task |
|---|---|
| Thu Oct 8 | T3.5 quality gates *(moved from Wed; T4.2b ran Oct 5–7)* |
| Fri Oct 9 | T4.5 `make demo` |
| **Sun Oct 11** | **T3.4 drift analysis: hand-label in one sitting** |
| Mon Oct 12 | README |
| Tue Oct 13 | T3.7 runbook · decide on the API key |
| Wed Oct 14 | T2.6 model card if T4.1 is skipped; otherwise T4.1 |
| Thu Oct 15 | online wiring (optional) or buffer |
| Fri Oct 16 | buffer (T2.6 here if T4.1 is built) |
| **Sun Oct 18** | **T4.4 Actions + T4.6 release** |

If I fall behind, drop T3.6 (Reddit) first and T3.4 last.

**Numbers later tasks depend on**

| | |
|---|---|
| Data | `sentences_75agree`, 3,453 rows · 70/15/15 stratified, seed 42 · test n=518, SHA-256 pinned · 12.2% negative · `max_length` 96 |
| Baselines (test: acc / macro-F1 / neg recall) | Majority 62.16 / 25.56 / 0.00 · Stratified 44.02 / 30.54 / 11.11 · VADER 54.05 / 45.83 / 22.22 · LogReg 83.59 / 79.97 / 77.78 |
| DistilBERT, **test** | 92.08% acc · 90.19% macro-F1 · recall neg/neu/pos 96.83 / 93.17 / 87.22 · 41 errors vs LogReg's 85. **The published model's number.** Val (93.44 / 90.79, epoch 3) is not comparable |
| FinBERT | 95.37 / 94.45: trained on PhraseBank, reference only, **not a ceiling** |
| Published model | `kkrit-tinna/newsmood-distilbert-financial` @ `be7b809e0d8f7bd50e77d902aee199363b7e2a66`, 269 MB |
| Throughput | 164.9 headlines/s on M3 Pro CPU, batch 32 · cold start 3.05 s (`docs/model_card.md`) |
| Feeds | cnbc-finance and marketwatch-top live. yahoo-finance stale since Sep 23, **HTTP 404 on Oct 7** |
| Live volume | ~15 headlines/day at one ingest a day (12–29 on ingest days, 1–4 otherwise) |
| Live label mix | Sep 29, n=112: 14.3% neg · 61.6% neu · 24.1% pos |
| T1.4 vectorizer OOV on live headlines | 36.9% token / 53.5% type over 147 headlines, so the default is the `window` vectorizer |
| Latest report | Oct 7: mood −9.4, n=13, up 17.0 from Oct 2. Re-run byte-identical. Cluster block declined, as on all six stored days with ≥ 8 headlines |

**Ready:** repo, env, HF token, PyPI name (`newsmood` / `newsmood-cli`).
**Open:** Anthropic API key. Optional, only needed for T4.1; decide by Oct 13.

**Quirks:** Apple Silicon picks `mps`; if training looks stuck, try `--device cpu`.
Run `newsmood ingest` several times a day until T4.4 automates it.

**Carry forward**, grouped by the session that acts on it

*T3.5 — gates (Thu Oct 8)*
- Hard blocks for broken runs (feeds down, dedupe failing, all items stale)
  versus disclosure for thin days. `min_headlines: 15` contradicts T3.3's
  no-minimum rule. T4.4's Done-when and §8's "Gates with teeth" depend on the
  outcome. `ReportData.gates` replaces "not yet implemented".
- Yahoo: stale, 404, and 29 of 49 items from one publisher. Likely drop.
  Optionally probe one replacement RSS feed (free, a different newsroom,
  fresh, not syndication-heavy, passes two live fetches), with a 20-minute time box.
- MarketWatch is 85% neutral (n=20), mostly advice columns.

*T4.5 — make demo (Fri Oct 9)*
- `report --input` is undesigned: sample rows have no url, source or
  `published_at`; `score --file` writes `pred_label`; the input path also needs
  `ThemeInputs` (background rows, `history_days`).
- Must not write to the store or `reports/` (use `report.output_dir`). The source
  "unknown" counts as one source in the header.
- The `daily_index` column list in "Using the data" is stale.

*T3.4 — drift (Sun Oct 11)*
- The hand labels settle whether the live label mix reflects correct labels or the
  model's prior.
- Labelled example: "Tesla sold a lot more EVs than Wall Street expected, and
  the stock is surging", labelled negative at 39% (Oct 2).

*README (Mon Oct 12)*
- `PYTORCH_ENABLE_MPS_FALLBACK=1`. Reports render fully only on GitHub
  (Mermaid pie, `<details>`).
- The cluster block declines at ~15 headlines/day; say why. Cluster names rank
  by raw count until 30 days after the first fetch (Oct 25).

*T2.6 — model card (Oct 14 or 16)*
- Epoch 3 shipped (val loss 0.1941; EarlyStopping never fired). Epoch 2's
  94.02% val accuracy is not the published number.
- FinBERT isn't zero-shot: fix §2, the T2.3 body, and decision 6 above.
- Top confusion is neutral→positive. Still open: DistilBERT against LogReg's
  four failure types on the same rows, lone nouns first.
- Trained on MPS: reproducible in substance, bit-exact only with `--device cpu`.
  Actions scores on CPU, local runs on MPS.
- State that dedupe drops the syndication count. Replace T2.5's "2020 laptop"
  example. The Hub README comes from `docs/model_card.md`; don't bump
  `model.revision` for a README-only commit.

*T4.4 — Actions (Sun Oct 18)*
- Ingest every 4–6 h, score and report once; update §2's diagram and the cron.
  Then re-check `explain.cluster_distance` (0.6) against the higher volume.
- Cache `~/.cache/newsmood`, keyed on `model.revision`.
- `daily_index` has real rows from Oct 1, plus thin frozen rows for Oct 3–5. Clear
  it before launch only if the trend should start at launch.

*Unassigned cleanup*
- §0.5 calendar is stale: Oct 5 still says "source skew"; dates have shifted.
- `perform_cross_validation()` is unported: assign it or drop it.
- T1.3 lexicon facts are wrong ("liability" −0.8; "aggressive growth" +0.25):
  guide edit + §9.
- `docs/baselines.md` VADER wording (~line 202) assumes one mechanism; the
  line-155 CLAIM splits the negative errors 23/15.
- Optional: one-line RSS summary per article (needs `&amp;` unescaping).
- Minor: `compute_mood` divides by zero (unreachable); `scored_at` uses the real clock.

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