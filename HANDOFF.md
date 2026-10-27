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

*Last updated: Tuesday, Oct 27, 2026*

**Status:** T4.4 and T3.4 done. `ingest.yml` (every 6 h) and `daily.yml`
(weekdays 22:00 UTC) run `--offline` on GitHub Actions; the store lives on
the orphan `data` branch (seeded Oct 26, 646 headlines), and both workflows
fetch it first and push it back. Manual dispatch checks passed, and the
overnight scheduled `ingest` runs went green. **Manual `newsmood ingest` has
stopped**: the local DB is no longer the store of record. T3.4 is committed:
`docs/drift.md` reports 59/100 live vs 92.08% in-domain, with six error
examples (one per off-diagonal cell). T1.1–T1.6, T2.1–T2.5, T3.1–T3.5, T4.2,
T4.3, T4.2b and T4.4 complete. 363 tests.

**Next:** T4.5, Wed Oct 28 (~45 min): `make demo` against a temp DB and
output folder via `NEWSMOOD_STORE__DB_PATH` / `NEWSMOOD_REPORT__OUTPUT_DIR`,
plus the wheel smoke test. `git pull` first; the bot commits a report to
main each weekday.

| Date | Task |
|---|---|
| ~~Mon Oct 26~~ | ~~T4.4 Actions~~ ✓ |
| ~~Tue Oct 27~~ | ~~T3.4 write-up + commit~~ ✓ |
| **Wed Oct 28** | **T4.5 `make demo` + wheel smoke test (~45 min)** |
| Thu Oct 29 | T2.6 model card + README, T3.7 folded in (~60 min) |
| **Fri Oct 30** | **T4.6 release ← end date (was Thu Oct 22)** |
| Oct 31 – Nov 1 | buffer |
| after release | PyPI publishing, unassigned cleanup, T4.1 |

T3.6 (Reddit) is dropped. T4.1 + online wiring are deferred until after release.

**Numbers later tasks depend on**

| | |
|---|---|
| Data | `sentences_75agree`, 3,453 rows · 70/15/15 stratified, seed 42 · test n=518, SHA-256 pinned · 12.2% negative · `max_length` 96 |
| Baselines (test: acc / macro-F1 / neg recall) | Majority 62.16 / 25.56 / 0.00 · Stratified 44.02 / 30.54 / 11.11 · VADER 54.05 / 45.83 / 22.22 · LogReg 83.59 / 79.97 / 77.78 |
| DistilBERT, **test** | 92.08% acc · 90.19% macro-F1 · recall neg/neu/pos 96.83 / 93.17 / 87.22 · 41 errors vs LogReg's 85. **The published model's number.** Val (93.44 / 90.79, epoch 3) is not comparable |
| FinBERT | 95.37 / 94.45: trained on PhraseBank, reference only, **not a ceiling** |
| T3.4 live headlines (`data/t34_analysis.json`) | Acc vs my labels: all 59/100 = **59.00%** [49.20, 68.13] · unflagged 49/80 = 61.25% [50.29, 71.18] · hard 10/20 = 50.00% [29.93, 70.07] (Wilson 95%), vs **92.08%** PhraseBank test (n=518) · recall neg 16/28 = 57.14 · neu 36/45 = 80.00 · pos 7/27 = 25.93 · my mix 28 neg / 45 neu / 27 pos, 20 hard · self-agreement 14/20 = 70.00%, κ 0.549 · largest cell positive→neutral, 15 (9 at confidence ≥ 0.85) · by source: marketwatch 32/47 = 68.09 · cnbc 12/22 = 54.55 · investing 15/31 = 48.39 |
| Published model | `kkrit-tinna/newsmood-distilbert-financial` @ `be7b809e0d8f7bd50e77d902aee199363b7e2a66`, 269 MB |
| Throughput | 164.9 headlines/s on M3 Pro CPU, batch 32 · cold start 3.05 s (`docs/model_card.md`) |
| Feeds | cnbc-finance, marketwatch-top, **investing-stocks** (added Oct 9; 10 items ≈ 20 min of news; naive dates, verified UTC). yahoo-finance removed Oct 9 (stale since Sep 23, 404 Oct 7–8); its stored rows stay. **Source mix changes from Oct 9** |
| Store (Oct 14) | 336 rows: marketwatch 113 · investing 96 · cnbc 78 · yahoo 49 (some from 2024). T3.4 pool (Oct 12): 168 scored rows published on or after the first fetch |
| Live volume | ~15 headlines/day at one ingest a day (12–29 on ingest days, 1–4 otherwise) |
| Live label mix | Sep 29, n=112: 14.3% neg · 61.6% neu · 24.1% pos |
| T1.4 vectorizer OOV on live headlines | 36.9% token / 53.5% type over 147 headlines, so the default is the `window` vectorizer |
| Latest report | Oct 9 (morning, partial day, not committed): mood −29.0, n=19, 3 sources, gates 2/2, no data notes. Oct 7: −9.4, n=13. Cluster block has declined on every stored day |

**Ready:** repo, env, HF token, PyPI name (`newsmood` / `newsmood-cli`).
**Open:** Anthropic API key. Only needed for T4.1, which is deferred until after release.

**Quirks:** Apple Silicon picks `mps`; if training looks stuck, try `--device cpu`.
Run `newsmood ingest` 2–3 times on weekdays (morning, midday, evening), at least once a day, until T4.4 automates it.

**Carry forward**, grouped by the session that acts on it

*T4.4 — Actions (Mon Oct 26)*
- DB branch plan: orphan `data` branch holds only `newsmood.db`, never merged.
  Seed it by hand from the local DB first. `ingest.yml` every 6 h and
  `daily.yml` weekdays 22:00 UTC both fetch the DB from `data` and commit it
  back; one `concurrency` group, `cancel-in-progress: false`. Fallback: only
  `daily.yml` today, `ingest.yml` later.
- Ingest every 6 h, score and report once; update §2's diagram and the cron.
  Then re-check `explain.cluster_distance` (0.6) against the higher volume.
- Cache `~/.cache/newsmood`, keyed on `model.revision`.
- `daily_index` has real rows from Oct 1, plus thin frozen rows for Oct 3–5. Clear
  it before launch only if the trend should start at launch.
- Done-when's forced failure: `NEWSMOOD_GATES__MAX_FEED_STALENESS_HOURS=0` →
  exit 1, `pipeline_fresh` named on stderr. Decide whether the workflow commits
  `reports/quality/*.json` (audit trail) or only the reports.
- Each ingest adds ~10 Investing.com items (≈ 20 min of its news), so its daily
  share depends on cadence. Check source shares after the first week.

*T3.4 — write-up (Tue Oct 27)*
- I pick the five errors and write the reasoning myself; Claude may draft the rest.
- Headline finding: positive recall is low, mostly positive→neutral at high
  confidence (deals, launches, "Why is X stock rallying today?"), cases rule 2
  settled in advance.
- Report self-agreement as a reliability limit. 4 of the 6 changed labels weren't
  flagged hard; the relabel flagged 7 vs 4 originally.
- One story is sampled twice (the two Amazon/Decart ids, `94a3bd4385dc` and
  `4973b35ad5f6`); say it, don't drop it.
- The drop exceeds the guide's 10–20 point expectation: a finding, not a §9 deviation.
- Commit set: `docs/drift.md`, `data/sample/hand_labeled_100.jsonl`,
  `scripts/t34_*.py`. `data/t34_*.jsonl` and `data/t34_analysis.json` stay
  gitignored.

*T4.5 — make demo (Wed Oct 28)*
- `report --input` is undesigned: sample rows have no url, source or
  `published_at`; `score --file` writes `pred_label`; the input path also needs
  `ThemeInputs` (background rows, `history_days`).
- Must not write to the store or `reports/` (use `report.output_dir`). The source
  "unknown" counts as one source in the header.
- The `daily_index` column list in "Using the data" is stale.

*T2.6 — model card (Thu Oct 29)*
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
- Near-duplicate titles of one story pass the `title_norm` dedupe (T3.4 sampled
  Amazon/Decart twice).

*README (Thu Oct 29)*
- `PYTORCH_ENABLE_MPS_FALLBACK=1`. Reports render fully only on GitHub
  (Mermaid pie, `<details>`).
- The cluster block declines at ~15 headlines/day; say why. Cluster names rank
  by raw count until 30 days after the first fetch (Oct 25).
- "When a run fails" (T3.7 folded in): a block exits 1, writes no report, names
  the gate on stderr, and leaves `reports/quality/{date}.json`. Exit 2 is a
  rejected request. A blocked run doesn't compute the index. Feed gates are
  `not_applicable` on a past `--date`. Backfill with `--date`; restore the DB
  from the `data` branch.

*T4.6 — release (Fri Oct 30)*
- Build the wheel, install it in a fresh venv, `cd /tmp`, run `newsmood --help`
  and `newsmood report --offline`. If `config/default.yaml` is resolved
  relative to the working directory, it won't ship: package it and load it
  with `importlib.resources`.

*Unassigned cleanup*
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
  age ≥ threshold. `reports/quality/{date}.json` every run, pass or fail.
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