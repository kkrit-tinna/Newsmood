# Week 3 Summary — Sep 28 – Oct 4, 2026

**Planned:** T2.4 (Mon), T2.5 (Tue), T3.3 (Wed), T4.2 (Thu), then T4.3,
the first offline report, on Friday.
**Delivered:** all five tasks. Thursday was lost, so T4.2 and T4.3 ran
together on Friday as two sessions with a commit between them. The report
layout was redesigned mid-task, and three design gaps surfaced and were
closed before they shipped.
**Net position:** on schedule. **The first offline report landed on its
target date, Oct 2**, built from live RSS with no API key. The Oct 9 slack
is untouched.

---

## Done

### T2.4 — Hub publish (Mon)

- Published to `kkrit-tinna/newsmood-distilbert-financial` (public) with
  `hf upload`, pinned in config at commit `be7b809e…`
- `models/loader.py` checks the cache first (`local_files_only`), downloads
  only on a miss with one stderr notice, uses
  `cache_dir=~/.cache/newsmood` and `token=False`, and never reads
  `models/local/`
- `config.py` rejects any revision that isn't a full 40-character SHA:
  `main`, branch names and short hashes all fail
- A thin `score --text` was pulled forward, because T2.4's Done-when
  needed it

**The label check.** `id2label` showed negative/neutral/positive by name,
but in alphabetical order, which is exactly the failure week 2 warned
about, since a correct mapping and an accidental one look identical.
Settled by running three unambiguous sentences through the model rather
than trusting the config.

**The Done-when passed for the wrong reason.** On a clean cache,
`score --text "Profit fell sharply"` printed `negative`. But torch and
transformers sat behind the `[train]` extra, so a plain `pip install`
couldn't score at all. The run passed only because the dev environment had
the extras. Fixed the same night: torch, transformers, datasets and tqdm
moved to core (`build: core-inference-deps`).

### T2.5 — Batched scoring (Tue)

- Three modes, decided before any code: bare `score` writes unscored SQLite
  rows (the only mode that writes); `--text` and `--file` are read-only;
  they never combine
- `--file` adds `pred_label`, not `label`, because the sample already
  carries the gold label and the prediction would have overwritten it
- One inference path: `classify()` wraps `classify_batch()`, and
  `eval --all` was rerun to confirm `docs/baselines.md` was unchanged
- Notices go to stderr, results to stdout, so output is safe to pipe

| Benchmark (M3 Pro, CPU, 5 threads, batch 32) | Value |
|---|---|
| Throughput | 164.9 headlines/s (~6 s per 1,000) |
| Batched latency p50 / p95 | 6.09 / 6.72 ms per headline (7 batches only) |
| Cold start | 3.05 s, mostly imports and model load |

Input was PhraseBank sentences, not live headlines, so live throughput is
likely higher. `docs/model_card.md` was started early, with this
Performance section only.

**First live scoring (n=112):** 14.3% negative, 61.6% neutral, 24.1%
positive. MarketWatch stands out at 85% neutral (n=20). Yahoo's positive
share (29%) matches CNBC's (28%): it's the largest source, not a skewed
one.

### T3.3 — Mood index (Wed)

Formula: `mood = 100 × Σ(conf·s) / Σ(conf)`, a confidence-weighted average
of signs.

Five decisions, each made from measured data rather than the spec:

| Decision | Evidence |
|---|---|
| Day = America/New_York, always | Under UTC, Sep 22–23 were 3 and 46 headlines; under ET, 29 and 21. About 26 headlines changed days |
| No 0.60 confidence cutoff; every row counts | The cutoff dropped 18% of rows, but 30% of positives against 13% of neutrals, so it shifted the mood |
| No minimum count; stats stored with every day | Suppressing thin days hides them. "+100 · 1 headline" discloses them |
| Hybrid recompute: today and yesterday recomputed, older days frozen | Late headlines arrive within about a day; after that, history stays stable |
| Only requested days computed; empty days not stored | Stale Yahoo items from 2024 must not create index rows |

Per-headline terms (`conf·s`) are stored with each day, so a frozen day's
top movers come from the same snapshot as its mood. A dry run on a backup
copy of the store showed the stats doing their job: Sep 27's −100 visibly
rests on one headline.

**A test that couldn't fail.** The DST case (03:30Z on Nov 2) lands on
Nov 1 under both −4 and −5, so it couldn't catch a hardcoded offset.
Claude Code added 04:30Z, where correct and broken code disagree.

### T4.2 + T4.3 — Report template and first offline report (Fri)

**Redesigned layout.** The guide's full headline table became:

- a plain-language header (mood, band, n, sources, how many the model was
  unsure about)
- a Mermaid pie with a text fallback
- "Recommended articles": top headlines by certainty per sentiment, linked
  to the originals
- a source table
- the full list, collapsed in `<details>`
- a footer with a certainty legend and a note that links go to the
  original publisher, some of which limit free articles

**Three gaps caught before commit:**

1. **Frozen days and late headlines.** The builder raised when a day's rows
   didn't match the stored terms, which would have made re-running an old
   date fail forever. It now filters to the frozen snapshot and discloses
   `n_late`.
2. **Pie colors by position.** Mermaid colors slices by order, so on a day
   with no negatives, neutral would have drawn in negative's color. Colors
   now follow the sentiment, with colorblind-safe vermillion, grey and
   blue. Verified in GitHub's preview, not just in tests.
3. **Overlap with T4.2b.** Within a sentiment, contribution equals
   confidence (|s| = 1), so T4.2b's top movers were the same list as
   Recommended articles. Block 1 shrank to a contribution-percentage
   column.

**Pipeline:** `reporting/pipeline.py` runs ingest → score → index →
full-row query → `build_report_data` → `render` → atomic write.
`report.output_dir` is in config so `make demo` can write elsewhere.
Online mode errors until T4.1.

**The milestone:** `reports/2026-10-02.md` contains 14 headlines
(7 negative, 5 neutral, 2 positive), built with `ANTHROPIC_API_KEY` unset.
Counts sum to n, and the header mood matches `daily_index`. The first
re-run hash differed; a diff confirmed new headlines had arrived between
runs, and a back-to-back re-run was byte-identical.

Test suite: 171 → 278 over the week.

---

## Spec changes

- `hf` replaces `huggingface-cli`; cache at `~/.cache/newsmood`; thin
  `score --text` in T2.4 (Sep 28)
- torch, transformers, datasets and tqdm moved to core; `[train]` kept,
  empty (Sep 28)
- `--device` flag; benchmark on CPU, other modes `auto`; `scoring.py` and
  `benchmark.py`; `pred_label` (Sep 29)
- T3.3: ET day boundary, no 0.60 cutoff, no minimum count, hybrid
  recompute, requested days only, terms stored (Sep 30)
- T4.2 and T3.5 dependencies changed to T3.3 (Oct 2)
- T4.2 layout redesigned; `lede_bands` reused for the header; links with
  publisher note; colorblind-safe pie; late-headline exclusion (Oct 2)
- T4.2b block 1 folded into Recommended articles (Oct 2)
- `reporting/pipeline.py`, full-row store query, today also recomputes
  yesterday, feed-failure tolerance, `report.output_dir` (Oct 2)
- CLAUDE.md trimmed to durable coding rules; column lists point to
  `store.py`; agents no longer commit or edit HANDOFF (Sep 30)

---

## Deferred

**Gates versus disclosure (T3.5, Oct 7).** `min_headlines: 15` and
blocking gates contradict T3.3's no-minimum decision. T4.4's Done-when and
§8's "Gates with teeth" depend on how this is resolved.

**`make demo` (T4.5, Oct 9).** `report --input` is undesigned. The sample
rows have no url, source or date, and every demo source would show as
"unknown."

**Model card facts (T2.6, Oct 8).** FinBERT isn't zero-shot (the §2 and
T2.3 text still say it is). The top confusion is neutral→positive. The
four failure-type check is still open. MPS versus CPU confidences differ
slightly. Don't bump the pin for a README-only Hub commit.

**Ingest frequency (T4.4, Oct 18).** One fetch captures about 3 hours of
MarketWatch. Ingest every 4–6 hours.

**Minor:** `compute_mood` divides by zero if every confidence is 0
(unreachable); `scored_at` uses the real clock.

---

## Open items

1. **Yahoo has been stale since Sep 23.** The index runs on two feeds.
   Keep, gate or drop it in T3.5
2. **The cloud-session credit** ($100) must be claimed by **Oct 7** and
   expires Nov 4. It suits the code-and-tests sessions (T4.2b, T3.5); keep
   live-data work local
3. **The API key** should be supplied only if online mode is built (T4.1,
   optional); decide before Oct 14
4. **PyPI name** is still unrecorded in HANDOFF, open since T1.1
5. **Sun Oct 11** is T3.4 *and* movierec's heavy session

---

## Recurring lessons

**A passing Done-when can test the wrong environment.** T2.4 passed
because the dev machine had `[train]` installed, a dependency no user
would have. The question isn't whether the command works, but whether it
works on what a stranger has.

**A test is only useful if it can fail.** The first DST case gave the same
answer under the right and wrong offsets. Pie-color tests checked the text
the template wrote, not what GitHub drew. Pick inputs where correct and
broken code disagree, and check rendering where rendering is the risk.

**New rules collide with old ones.** Dropping the confidence cutoff turned
T3.5's gates into a contradiction. Freezing old days turned a strict
consistency check into a permanent failure. The cheap moment to find these
is when the rule is written, by asking which earlier decisions it touches.

**Files agents read must stay current.** CLAUDE.md still said "aggregate
replaces its row" after the hybrid rule made that false. A stale
instruction is worse than a missing one, because the next session follows
it.

---

## Week 4 starts here

**Mon Oct 5 – Tue Oct 6:** T4.2b, the Why section. Contribution %, source
skew and the lede, then distinctive terms and clusters. Decide where the
TF-IDF vectorizer lives for the Actions runner.

**Wed Oct 7:** T3.5 gates, the disclosure redesign. **Thu Oct 8:** T2.6
model card. **Fri Oct 9:** T4.5 `make demo`.

**Sun Oct 11: T3.4 drift analysis**, ~90 minutes in one sitting. The task
to protect above all others.