# Week 2 Summary — Sep 21–27, 2026

**Planned:** T1.4 through T1.6 (Mon–Wed), T3.1 and T3.2 (Thu–Fri), then the
fine-tune on Sunday.
**Delivered:** all seven tasks. T1.6's review pass ran long enough to eat
Wednesday night, and a threshold-sweep script that never existed had to be
written before its own table could be trusted.
**Net position:** ahead. Phase 1 is complete, Phase 3's ingest and storage
are done, and DistilBERT is fine-tuned and evaluated. The fine-tune took 66
seconds, not the 20–40 minutes the guide budgeted.

---

## Done

### T1.4 — TF-IDF + Logistic Regression (Mon)

- `preprocessing.py` kept minimal: lowercase, collapse whitespace, strip
  URLs and tickers. Docstring says TF-IDF only, never in front of
  DistilBERT or VADER
- Vectorizer and classifier persisted separately to `models/baselines/`
  (no `Pipeline`), so `reporting/explain.py` can reuse the vectorizer alone
  in T4.2b. `meta.json` records sklearn version, train row count and the
  split hash
- Leakage test uses a sentinel token seeded into ≥2 test rows and zero
  train rows, so it would survive `min_df=2` if it leaked. It does not
  appear in `vectorizer.vocabulary_`

| Method | Accuracy | Macro-F1 | Weighted-F1 | Negative recall |
|---|---|---|---|---|
| Majority (neutral) | 62.16% | 25.56% | 47.66% | 0.00% |
| LogReg | 83.59% | 79.97% | 83.44% | 77.78% |

**The missing dependency.** `scikit-learn` was never declared in
`pyproject.toml`. Added as a **core** dependency rather than under
`[train]`, because `explain.py` loads the persisted vectorizer during
`newsmood report --offline`.

### T1.5 — metrics and the floor rows (Tue)

- `calculate_comprehensive_metrics` returns a plain dict keyed by method
  name, so T2.3 extends it rather than replacing it
- `baselines/reference.py` adds majority-class and stratified-random
- `eval --all` wired; `baselines_doc.py` splices the table into
  `docs/baselines.md` between markers
- Stratified random: 44.02% accuracy, 30.54% macro-F1. Below the floor on
  accuracy, above it on macro-F1

**The doc-writer deviation.** The guide implies `eval --all` generates the
whole file. It does not — the file already carried hand-written analysis
from T1.3. Scoped to a marker-delimited block instead of a whole-file
overwrite, and verified idempotent.

### T1.6 — baseline analysis (Wed, plus a long review)

Drafted with a `CLAIM:` marker convention: measured statements as plain
prose, interpretive ones wrapped in HTML comments for review. Twelve
claims. Two were substantively wrong, two overstated, eight stood.

**Claim 1 was wrong.** It asserted direction blindness, *not* missing
vocabulary, explains negative being VADER's worst class. Measured, of 49
negative errors:

| Bucket | Count |
|---|---|
| Positive noun + direction word → predicted positive | 23 (all 23) |
| Zero lexicon hits → compound 0 → neutral | 15 (all 15) |
| Predicted positive, no listed direction word | 9 |
| Hits nearly cancelling, inside the band | 2 |

Split into two mechanisms: a compositional failure and a coverage failure.

**The sweep script did not exist.** Three of four rows in a committed table
came from an inline script that was never saved. Rewritten as
`scripts/vader_threshold_sweep.py`; all four rows reproduce exactly. Of 780
valid pairs, 102 clear the floor, two edge past the configured macro-F1 by
0.28 and 0.15 points, and **no pair with a negative cutoff below zero beats
22.22% negative recall**. The 64.48% best-accuracy row was a three-way tie
showing the weakest member; corrected to +0.70/−0.30.

**Two factual corrections.** "Loss narrowed" reads *neutral* (−0.0258), not
negative — `HANDOFF.md` and `week1_summary.md` both said negative. And the
221 zero-compound sentences are not unrecoverable by any threshold; at
least 44 of them are wrong under every threshold.

### T3.1 — RSS ingest (Thu)

Each feed fetched twice, 20 minutes apart. Counts matched.

| Feed | Entries | Coverage | Verdict |
|---|---|---|---|
| yahoo-finance | 49 | newest item 43h old; 29 of 49 from one publisher; no summaries | live but stale, kept |
| cnbc-finance | 30 | ~2 weeks | kept |
| marketwatch-top | 10 | ~3 hours | kept |

89 rows per fetch, but only **16 published on Sep 24**. `data/feeds.py`
fetches sequentially with a descriptive User-Agent, 15s timeout and a 2s
pause; a dead feed becomes an error in its result rather than stopping the
run. T3.1's dependency corrected from T2.5 to T1.1.

### T3.2 — SQLite store (Fri)

Four decisions, all recorded:

1. `id` stays the URL hash — it makes re-ingest idempotent
2. `UNIQUE(title_norm)` — the rule lives in the schema, not in one code path
3. On conflict, first-seen row wins; `published_at` may move **earlier**
   only; `fetched_at` never moves
4. Known limitation logged for the T2.6 model card: collapsing five outlets
   into one row discards the syndication count

`title_norm` unescapes HTML entities, normalizes curly quotes and dashes,
and maps punctuation to a space rather than deleting it, so "2.5%" and
"25%" stay distinct. Yahoo moved last in the feed list, since first-seen
wins and Yahoo has no summaries.

Live feeds produced 89 headlines with 89 distinct `title_norm` values —
**zero natural collisions**, so the cross-feed test case was constructed
with two fake outlets rather than assumed.

### T2.1 + T2.2 + T2.3 — the fine-tune (Sun)

**`max_length` 96, not 58.** Measured p99 is 69. The p95 of 58 would have
truncated 159 of 3,453 rows; 96 truncates 3. With `DataCollatorWithPadding`
the higher cap costs nothing, since batches pad to their own longest row.

**Two bugs caught before the run.** `EarlyStopping.save_checkpoint` used
`state_dict().copy()`, a shallow copy whose tensors alias the live
parameters — "best" weights were overwritten by every optimizer step. Fixed
with `deepcopy` and a test that fails against the original line. And typer
silently dropped `click_type`, so `--device` would have accepted any value;
the CLI choice-list test caught it.

**The run:** 456 steps, 45 warmup, MPS, seed 42.

| Epoch | Train loss | Val loss | Val acc | Val macro-F1 | Time |
|---|---|---|---|---|---|
| 1 | 0.7467 | 0.3164 | 88.61% | 84.96% | 24.9s |
| 2 | 0.2326 | 0.2128 | 94.02% | 91.16% | 21.4s |
| 3 | 0.1286 | 0.1941 | 93.44% | 90.79% | 20.7s |

**66 seconds total, against a budgeted 20–40 minutes.** The old estimate
assumed padding to 192 tokens. Saved weights are epoch 3, selected on val
loss; epoch 2 had the higher val accuracy.

**Test split, six rows:**

| Method | Accuracy | Macro-F1 | Weighted-F1 | Recall neg / neu / pos |
|---|---|---|---|---|
| Majority class | 62.16% | 25.56% | 47.66% | 0.00 / 100.00 / 0.00 |
| Stratified random | 44.02% | 30.54% | 43.99% | 11.11 / 60.25 / 20.30 |
| VADER | 54.05% | 45.83% | 55.29% | 22.22 / 54.97 / 66.92 |
| Logistic Regression | 83.59% | 79.97% | 83.44% | 77.78 / 90.06 / 70.68 |
| **DistilBERT (fine-tuned)** | **92.08%** | **90.19%** | **92.16%** | **96.83 / 93.17 / 87.22** |
| FinBERT (reference only) | 95.37% | 94.45% | 95.42% | 98.41 / 94.10 / 96.99 |

**FinBERT is not a clean ceiling.** `ProsusAI/finbert` was fine-tuned on
Financial PhraseBank, and the `75agree` test sentences sit inside the larger
configs it saw. The released checkpoint does not ship its training split, so
the overlap cannot be measured. Row relabelled from "zero-shot" to "trained
on PhraseBank, reference only." Its 98.41% negative recall — the hardest,
scarcest class — is the tell.

**The guide's confusion prediction was wrong.** It expected
negative→neutral; only 2 negatives were misread that way. The dominant error
is **neutral→positive (17)** — the same single-noun false positive that
defeated VADER and survived LogReg. Nine positives were read as negative,
the direction that moves a mood index most. Negative precision is 61/75 =
81%, consistent with the ×2.74 class weight.

The real headline: **negative recall 96.83% against LogReg's 77.78%**, on a
class with 294 training examples.

---

## Spec changes

- `scikit-learn` added as a core dependency (Sep 21)
- `eval --all` writes a marker-delimited block, not the whole file (Sep 22)
- `scripts/vader_threshold_sweep.py` added; best-accuracy row corrected to
  the macro-F1-best member of a three-way tie (Sep 24)
- T3.1 dependency T2.5 → T1.1 (Sep 24)
- `yahoo-finance` moved last in the feed list (Sep 25)
- `max_length` 96 with dynamic padding, superseding the measured p95 of 58
  (Sep 27)
- `EarlyStopping.save_checkpoint` fixed rather than ported as-is (Sep 27)
- T2.1 Done-when gains `--device cpu`; `get_device()` reordered
  cuda > mps > cpu (Sep 27)
- `tqdm` declared in the `[train]` extra (Sep 27)
- T2.2 runtime estimate corrected from 20–40 min to ~66s (Sep 27)
- T2.3 Done-when: five rows → six (Sep 27)

---

## Deferred

**`perform_cross_validation()`** — carried three sessions, then dropped. Add
it only if a later margin is small enough that variance matters.

**The four failure types** — object-dependent direction, boilerplate
outvoting direction, lone financial nouns, negation scope. Still to be run
against DistilBERT on the same test rows. Given neutral→positive is the top
error, the lone-noun case is the one to check first.

**`&amp;` in MarketWatch summaries** — `title_norm` unescapes titles only.
Matters when T3.4 hand-labels or T4.2b surfaces summary text.

**`models/local/` is gitignored**, and `eval --all` now needs it. The
Actions runner will not have it until T2.4 publishes to the Hub. Same shape
as the T4.2b vectorizer question due Oct 5.

**T2.5 must reuse `resolve_device()`** — nearly nobody runs `newsmood
train`, so `score` is where portability actually matters, and it fails
quietly there.

---

## Open items

1. **16 headlines on Sep 24.** T3.3's index averages over the day. Decide
   the minimum count below which the index is not reported, alongside the
   existing confidence-dropout count
2. **Model card facts to carry:** published weights are epoch 3 selected on
   val loss; the run was MPS, so results are reproducible in substance but
   not bit-identical; FinBERT is not a ceiling
3. **PyPI name** — still not written into `HANDOFF.md`, open since T1.1
4. Sun Oct 11 is T3.4 *and* movierec's second heavy session, as Sep 27 was

---

## Recurring lessons

**A number in a doc needs committed code behind it.** Three of four rows in
the sweep table came from a script that no longer existed. Reproducing them
took ten minutes and turned up a three-way tie the doc had resolved in its
own favour. Ad hoc terminal work is fine until the result gets written down.

**The marker convention earns its cost, and has one.** Twelve claims, two
substantively wrong, both caught before they became prose to defend. But the
review ran past 1am because every marker invited a new measurement. The
judgment is which claims have downstream consequences — that was two of
twelve.

**Verify mappings against the source, not the code.** Checking
`LABEL_NAMES` against `id2label` only proves the code agrees with itself.
Printing four raw sentences per class is what would have caught an
alphabetically-assigned mapping.

**Estimates built on superseded assumptions stay wrong quietly.** The 20–40
minute training budget was the reason Sep 27 was declared unmovable. It
assumed 192-token padding, which stopped being true on Sep 17.

---

## Week 3 starts here

**Mon Sep 28:** T2.4, Hub publish. Token is created and saved; pin the
revision in config rather than tracking `main`.

**Tue Sep 29:** T2.5, batched scoring. **Wed Sep 30:** T3.3, mood index —
see open item 1. **Thu Oct 1:** T4.2, report template.

**Fri Oct 2: T4.3, the first offline report.** The milestone the schedule
was built around.

Sun Oct 4 is off.