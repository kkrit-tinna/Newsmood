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

*Last updated: Friday, Oct 2, 2026*

**Status:** T1.1–T1.6, T2.1–T2.5, T3.1–T3.3, T4.2 and T4.3 complete.
**Milestone reached: first offline report** (`reports/2026-10-02.md`),
on the target date, with no API key.

**Numbers later tasks depend on**

| | |
|---|---|
| Training config | `sentences_75agree`, 3,453 rows |
| Split | 70/15/15 stratified, seed 42 → 518-row test set, SHA-256 pinned in `splits_meta.json` |
| Negative share | 12.2% |
| Token length | mean 30.4 · p95 58 · p99 69 · max 150 → `max_length` 96, dynamic padding. Truncates 3 of 3,453 rows (0 in test) |
| Majority class (floor) | 62.16% acc · 25.56% macro-F1 · 0.00% negative recall — test split, n=518. (62.1% elsewhere is the same share over the full 3,453-row corpus, not this baseline — see `docs/dataset.md`.) |
| Stratified random | 44.02% acc · 30.54% macro-F1 · 11.11% negative recall — single seeded draw, `dataset.seed=42`, not averaged over repeats |
| VADER baseline | 54.05% acc · 45.83% macro-F1 · 22.22% negative recall |
| LogReg baseline | 83.59% acc · 79.97% macro-F1 · 83.44% weighted-F1 · 77.78% negative recall |
| DistilBERT — **TEST split** | 92.08% acc · 90.19% macro-F1 · 92.16% weighted-F1 · recall neg/neu/pos 96.83 / 93.17 / 87.22% — n=518, scored on CPU by `newsmood eval --all`. 41 errors vs LogReg's 85. This is the published model's number |
| DistilBERT — VALIDATION split | 93.44% val acc · 90.79% val macro-F1 — epoch 3 of 3, MPS, seed 42, n=518 val rows (`models/local/history.json`). Not comparable to the test rows |
| FinBERT | 95.37% acc · 94.45% macro-F1 — trained on PhraseBank, reference only, **not a ceiling** |
| Published model | `kkrit-tinna/newsmood-distilbert-financial` @ `be7b809e0d8f7bd50e77d902aee199363b7e2a66` (public). 269 MB, 7 files. `model.repo_id`/`model.revision` in config |
| RSS feeds | 3/3 live, in config order cnbc-finance 30, marketwatch-top 10, yahoo-finance 49 entries per fetch. Only 16 published on the T3.1 probe's US date. Yahoo lags about 43 h. See `docs/sources.md` |
| Store, first live ingest | Sep 25: 89 rows, 89 distinct `title_norm`, 89 with `published_at`. No cross-outlet duplicates. Re-run: 0 inserted, 89 duplicates |
| Human ceiling | qualitative only — up to 25% of annotators disagreed on every kept `75agree` row, no number fabricated |
| Throughput (T2.5) | 164.9 headlines/s on Apple M3 Pro **CPU**, batch 32, 5 threads (~6 s per 1,000). p50/p95 6.09/6.72 ms per headline batched (7 batches only). Cold start 3.05 s, mostly imports and model load. Input: PhraseBank sentences, not live headlines. See `docs/model_card.md` |
| Live store, first scoring | Sep 29: 112 rows, all scored on MPS. 14.3% negative · 61.6% neutral · 24.1% positive. By source (neg/neu/pos): cnbc 9/22/12, marketwatch 2/17/1, yahoo 5/30/14 |
| First offline report (T4.3) | Oct 2: 14 headlines, negative/neutral/positive 7/5/2, mood <MOOD>. Built from live RSS with ANTHROPIC_API_KEY unset. A back-to-back re-run was byte-identical, label counts sum to n, and the header mood matches `daily_index`. `daily_index` holds real rows from Oct 1–2 onward |

**Next task:** T4.2b, Why section part 1, Mon Oct 5.

**Schedule:** no sessions Sunday Sep 20 or Sunday Oct 4. Three Sundays are load-bearing and cannot move to a weekday: **Sep 27** (T2.1+T2.2+T2.3, the fine-tune), **Oct 11** (T3.4, hand-labeling in one sitting), **Oct 18** (T4.4 Actions + release). Full calendar in `IMPLEMENTATION_GUIDE.md` §0.5. If I fall behind, drop T3.6 (Reddit) first and T3.4 (drift analysis) last.

**Ready**
- [x] GitHub repo `newsmood`, public
- [x] Repo cloned, `newsmood_env` created, `pip install -e ".[train]"` working
- [x] PyPI name recorded: `newsmood` or `newsmood-cli`
- [x] Hugging Face account + write token — used for T2.4 upload (Sep 28)
- [ ] Anthropic API key — *deferred, optional.* Only for T4.1 online mode. Claude Pro does not include API access.

**Known environment quirks**
- Apple Silicon: `get_device()` returns `mps`. MPS is occasionally slower than CPU for small batches — if fine-tuning looks stuck, try `--device cpu` before debugging anything else.

**Carry forward** — grouped by the session that acts on it

*T4.2b — Why section (Mon Oct 5, Tue Oct 6)*
- Block 1 is a contribution % column on Recommended articles, not a new
  section. Ids are already carried in `ReportData`; contribution =
  conf / Σ|conf·s| over non-neutral headlines, from `daily_index.terms`.
  The lede replaces the "What happened" placeholder; source skew is one
  line under the Sources table.
- Stale guide text to fix: counterweight "above threshold" (no threshold
  exists); "~47 headlines a day" (~15 measured).
- `/models/` is gitignored, so the Actions runner won't have the fitted
  TF-IDF vectorizer (`models/baselines/`). Decide: refit in CI from the
  pinned split, commit the joblib file, or publish to the Hub.
- Optional: one-line RSS summary under each article; needs `&amp;`
  unescaping first (MarketWatch summaries).

*T3.5 — Quality gates (Wed Oct 7)*
- Gates vs disclosure: `min_headlines: 15` and blocking gates conflict with
  the T3.3 decisions (no minimum count, show stats instead). Decide hard
  blocks for broken runs (feeds down, dedupe failing, all items stale)
  versus disclosure for thin days. T4.4's Done-when (`min_headlines: 9999`)
  and §8's "Gates with teeth" depend on the outcome. `ReportData.gates`
  replaces "not yet implemented".
- Yahoo is stale: nothing newer than Sep 23 as of Sep 30, plus items from
  2024. The index currently runs on CNBC and MarketWatch only. Decide
  whether Yahoo stays, is gated, or is dropped.
- Live distribution, first look (n=112, Sep 29): MarketWatch is 85% neutral
  (n=20), the one source that stands out. Yahoo's positive share (29%)
  matches CNBC (28%); it's the largest source, not a skewed one.

*T2.6 — Model card (Thu Oct 8)*
- Epoch 3 shipped, selected on val loss (0.1941). EarlyStopping never fired.
  Epoch 2 had the higher val accuracy (94.02% vs 93.44%); never read 94.02%
  as the published model's number.
- FinBERT is not a ceiling: trained on PhraseBank, overlap with our test set
  unmeasurable. Label it "trained on PhraseBank, reference only". §2 and the
  T2.3 body still say "zero-shot reference ceiling"; fix both.
- The top confusion is neutral→positive, not negative→neutral as T2.3
  predicts. Still open: check DistilBERT against LogReg's four failure types
  on the same test rows (object-dependent direction, boilerplate, lone
  financial nouns, negation scope), lone nouns first.
- Reproducibility: trained on MPS. Seeds fixed, but results are reproducible
  in substance, not bit-identical; exact reproduction needs `--device cpu`.
- Device split: local daily scoring runs on MPS, Actions on CPU, so
  confidences can differ slightly. Published reports are scored on CPU.
- Dedupe discards the syndication count (`UNIQUE(title_norm)`). Accepted
  trade; state it.
- `model_card.md` already has its Performance section (T2.5): ~165
  headlines/s on M3 Pro CPU. Replace the T2.5 body's "2020 laptop" example.
- Hub README is a placeholder in the gitignored `models/local/README.md`.
  T2.6 replaces it from `docs/model_card.md`. Lean: do NOT bump
  `model.revision` for a README-only commit (weights unchanged), so every
  report cites one SHA for one model.

*T4.5 — make demo (Fri Oct 9)*
- `report --input` is undesigned. Sample rows have no url, source or
  published_at; `score --file` writes `pred_label`, not `label`.
- The demo must not write to the store or `reports/`; point it elsewhere via
  the output-dir override.
- In the demo every row's source is "unknown", which the header counts as
  a source ("from 1 source").
- `daily_index` column list in "Using the data" is stale (missing
  `mean_conf`, `source_counts`, `terms`, `model_id`).

*T3.4 — Drift analysis (Sun Oct 11)*
- Whether the live split's resemblance to PhraseBank base rates reflects
  correct labels or the model echoing its prior: the hand labels settle it.

*README (Mon Oct 12)*
- Training section: `PYTORCH_ENABLE_MPS_FALLBACK=1` routes unimplemented MPS
  operators to CPU.
- Reports render fully only on GitHub (Mermaid pie, `<details>`); local
  previews show the pie as code, with the text line as fallback.

*T4.4 — Actions (Sun Oct 18)*
- Ingest every 4–6 h, score and report once: one fetch captures ~3 h of
  MarketWatch, so ingest days are dominated by its 10-item burst. Update
  §2's diagram and the cron.
- Cache `~/.cache/newsmood`, keyed on `model.revision`.
- `daily_index` holds real rows from Oct 1–2. Clear it before launch only
  if you want the stored trend to start at launch.

*Unassigned cleanup*
- `perform_cross_validation()` is unported; §3's reuse table maps it to
  `evaluation/metrics.py`, but no task needs it. Assign it or drop it.
- Lexicon facts in T1.3 prose are partly wrong: "liability" is −0.8
  (compound −0.2023 alone); "aggressive growth" scores +0.25. Guide edit
  + §9 line.
- `docs/baselines.md` VADER wording: the "Why no threshold fixes either"
  heading and "the class direction blindness breaks" (~line 202) assume one
  mechanism; the line-155 CLAIM splits negative errors 23/15. Update both.
- Minor: `compute_mood` divides by zero if every confidence is 0
  (unreachable, floor is 1/3); `scored_at` uses the real clock while
  `fetched_at` uses the injected one. Neither affects reports.

**Reference — built components and invariants**
- **Timestamps**: `YYYY-MM-DDTHH:MM:SS.ffffff+00:00`, 32 chars, UTC, via
  `store._ts` only. Day ranges are string comparisons and depend on it.
- **T1.5**: `eval --all` writes the summary table between
  `<!-- eval:summary:start/end -->` in `docs/baselines.md`.
- **T1.6**: interpretive sentences in `docs/baselines.md` §Analysis are
  wrapped in `<!-- CLAIM: ... -->` (`grep CLAIM:`).
- **T2.4**: `loader.ensure_model()` checks the cache (`local_files_only`),
  downloads only on a miss with one stderr notice, pinned SHA,
  `cache_dir=~/.cache/newsmood`, `token=False`. Never reads `models/local/`.
  `config.py` rejects any revision that isn't a full 40-char SHA.
- **T2.5**: bare `score` writes only `label IS NULL` rows (never overwrites);
  `--text`/`--file` are read-only; `--file` reads `sentence`, adds
  `pred_label` + `confidence`. `--benchmark` needs `--file`, defaults to CPU;
  other modes `auto`. `model.batch_size: 32`. Code in `models/scoring.py`,
  `models/benchmark.py`.
- **T3.2**: `store.py` owns the schema. Dedupe `UNIQUE(title_norm)`, first
  seen wins; `published_at` only moves earlier. Label/confidence/scored_at
  are all set or all NULL. Index buckets by `published_at`.
- **T3.3**: ET day; every row counts, weighted by confidence; no minimum;
  stats stored per day; today and yesterday recomputed, older days frozen;
  empty days not stored; terms stored for T4.2b.
- **T4.2/T4.3**: `report_data.build_report_data()` (all arithmetic) →
  `template.render()` (formatting only) → `pipeline.run_report()` writes
  `reports/{date}.md` atomically. Late headlines for a frozen day are
  excluded and counted (`n_late`). Re-runs are byte-identical.

**Deviations so far** — full entries in `IMPLEMENTATION_GUIDE.md` §9
- `.gitignore`: `data/` → `data/*`, `models/` → `/models/`
- `datasets>=4.0` dropped script loading → raw zip download, SHA-256
  pinned, `PhraseBankSourceChanged` on mismatch
- `max_length`: placeholder 192 → p95 58 → 96 with dynamic padding
- `pytest` as a `dev` extra; scikit-learn, `huggingface_hub>=1.32,<2`, torch,
  transformers, datasets and tqdm are core. `[train]` is empty
- `config.py` precedence: explicit > env > yaml > default
- All three feeds kept; Yahoo moved last; `store.db_path` in config
- `eval --all` writes only the marker-delimited table
- `EarlyStopping` shallow-copy bug fixed; `get_device()` cuda > mps > cpu
- T2.2: 67 s on MPS. T2.3: FinBERT relabelled; Done-when six rows
- T2.4: `hf upload`; cache `~/.cache/newsmood`; thin `score --text` in T2.4
- T2.5: `--device`; benchmark on CPU; `scoring.py`/`benchmark.py`;
  `model.batch_size`; `pred_label`; `model_card.md` started early
- T3.3: ET day; no 0.60 cutoff; no minimum count + stats; hybrid recompute;
  requested days only; empty days not stored; terms stored
- T4.2/T3.5: dependencies changed to T3.3. T4.2: redesigned layout (pie,
  recommended articles with links, source table, collapsed list),
  `lede_bands` reuse, colorblind-safe pie colors, late-headline exclusion,
  gates placeholder, `model_id` printed once
- T4.2b: block 1 folded into Recommended articles
- T4.3: `reporting/pipeline.py`; full-row store query; today also recomputes
  yesterday; feed-failure tolerance; online mode errors until T4.1; 
  Move the reports directory to config as report.output_dir (default
  "reports"), read through settings like store.db_path. Keep the injectable
  override for tests. Add the key to test_config's throwaway yaml and note it
  in the existing §9 row for the pipeline module.

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