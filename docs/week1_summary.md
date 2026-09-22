# Week 1 Summary — Sep 14–18, 2026

**Planned:** finalize the plan (Mon–Tue), then T1.1 through T1.3 (Wed–Fri).
**Delivered:** all three tasks, plus two hardening follow-ups the guide did
not ask for: a checksum on the raw dataset source, and a fix to config
precedence.
**Net position:** on schedule. Phase 1 is 3 of 6 tasks done, and the numbers
every later task depends on (the 62.1% floor, the pinned test split, the
token-length distribution) are measured rather than estimated.

---

## Before the first commit (Mon–Tue)

The plan changed shape before any code was written:
- **Offline-first.** The Claude API became optional. `newsmood report
  --offline` produces the full report, and the project costs $0.00 in
  that mode
- **Computed explanation (T4.2b).** A template lede sentence plus top
  movers in both directions, distinctive terms, clusters and source skew,
  all in Python, so the offline report is readable on its own
- **Schedule set to Sep 16 – Oct 18.** Sundays Sep 20 and Oct 4 off. Ingest
  moved ahead of the model since it has no dependency on it, which puts
  the first offline report on Fri Oct 2
- **Data sources settled:** Financial PhraseBank for training, RSS for
  daily input. Other corpora were evaluated and dropped

---

## Done

### T1.1 — package skeleton (Wed)
- `.gitignore` as commit #1. `newsmood_env/` ignored explicitly, since a
  venv without a leading dot is the easiest thing to commit by accident
- `pyproject.toml`: typer, feedparser and pydantic-settings as core;
  torch, transformers and datasets behind a `[train]` extra. Entry point
  `newsmood = "newsmood.cli:app"`
- `CLAUDE.md` in the root before the skeleton, so module boundaries were
  decided before files existed
- Seven packages per §3; `cli.py` with five stubs printing
  `not implemented`
- `config/default.yaml` and `tests/__init__.py` added before the commit, so
  T1.2 had a place to put the seed instead of hardcoding it

**The .gitignore bug.** `data/` plus `!data/sample/` does not work: git
will not descend into an excluded directory to evaluate a negation.
Claude Code caught it with `git check-ignore -v` before anything depended
on it. Fixed to `data/*`.

### T1.2 — dataset and splits (Thu)
- `[train]` extra installed up front, so the fine-tune Sunday does not
  start with a 2 GB download
- Nesting verified by set containment on sentence text, not row counts.
  Zero violations
- Split 70/15/15, stratified on both peels. Negative held within 0.1pp
  across all three splits
- Pinned rather than just seeded: `splits_meta.json` records config,
  ratios, seed, `datasets` version and a SHA-256 of the sorted test
  sentences. `load_splits()` fails loudly if files are missing;
  `verify_splits()` raises on hash mismatch
- `data/sample/headlines_200.jsonl` drawn from the test split only

**What the data actually looks like:**

| | |
|---|---|
| Config row counts | 2,264 / 3,453 / 4,217 / 4,846, matching the guide exactly |
| Training config | `sentences_75agree`, 3,453 rows |
| Test split | 518 rows |
| Majority class | neutral, **62.1%** |
| Negative share | 12.2% |
| Token length | mean 30.4 · p95 58 · max 150 |

**The loading break.** `datasets>=4.0` removed script-based loading, and
this dataset never got a Parquet conversion, so `load_dataset()` fails.
Fixed by downloading the raw zip and parsing it with the original script's
own logic. Same data. The zip is unversioned upstream, so a follow-up
commit pins its SHA-256 and raises `PhraseBankSourceChanged` rather than a
`KeyError` on a checksum mismatch, a missing member, or a malformed line.

Test suite: 12 passing, all offline.

### T1.3 — VADER baseline (Fri)
- `vaderSentiment`, not nltk. The lexicon is bundled, so `make demo` cannot
  hit a runtime download or a `LookupError`
- Raw text to VADER. It keys on capitalization and punctuation, so cleaning
  would cripple the baseline and flatter the model later
- `evaluation/metrics.py` built on a shared `per_class_stats` core, shaped
  for T1.5 to extend rather than replace
- `cli.py`'s `eval` only dispatches on `--model`

**The result:**

| Method | Accuracy | Macro-F1 | Weighted-F1 | Negative recall |
|---|---|---|---|---|
| Majority (neutral) | 62.1% | — | — | 0% |
| VADER | 54.05% | 45.83% | 55.29% | 22.22% |

VADER loses to the floor on accuracy by 8 points. It beats the floor on
macro-F1, since always-neutral scores zero on two of three classes (about
25.5% by arithmetic; T1.5 puts the measured figure in the table). That
split between the two metrics is the reason both get reported.

Largest confusion cell: true neutral, predicted positive, 126 of 518 rows.

**Two failure modes, both structural:**
1. *Direction blindness.* The lexicon scores "profit" and "loss" as
   fixed-polarity words and cannot read the direction word beside them.
   "Loss narrowed" reads negative; "profit decreased" reads positive
2. *Single-noun false positives.* Neutral sentences whose only lexicon hit
   is a finance noun cross the +0.05 threshold: "shares" (0.1531), "profit"
   (0.4404), "growth" (0.3818)

Neither can be fixed by tuning a threshold. Both need a model that reads
words in context, which is the case for fine-tuning in one paragraph.

**The config bug.** `config.py` documented explicit > env > yaml >
default, but the yaml source outranked explicit constructor arguments.
T1.2's tests never exercised it because they passed `Path` arguments
directly. It surfaced when `predict_test_split(settings)` became the first
function to resolve a path from settings the way the CLI will, and
silently loaded the real 518-row split instead of the test fixture. The
first fix was half right, leaving env above explicit arguments; caught on
review and corrected the same day. The order is now asserted pairwise in
`tests/test_config.py`.

Test suite: 28 passing at the T1.3 commit, with `test_config.py` added in
the same-day follow-up.

### Spec changes
- venv named `newsmood_env` (Sep 16)
- `.env.example` moved from T1.1 to the optional T4.1 (Sep 16)
- `.gitignore`: `data/` → `data/*` (Sep 16)
- `[train]` install moved from Sep 27 to T1.2 (Sep 17)
- `max_length` placeholder 192 superseded by measured p95 of 58; the final
  value is decided in T2.1 (Sep 17)
- `pytest` declared as a `dev` extra; no task had added it
- `config.py` precedence corrected and tested (Sep 18)

---

## Deferred

**T2.1:** measure p99 token length before setting `max_length`. Choose 96
or 128. Not 64, which truncates the long tail, and financial sentences
often carry their sentiment in the final clause.

**T2.4 (Mon Sep 28):** needs a Hugging Face account and write token, not
yet created.

**T1.4:** fit the TF-IDF vectorizer once, on the training split, and
persist it. `reporting/explain.py` reuses it in T4.2b.

**`make demo`:** the 200-row sample is analyst sentences, not news
headlines. T3.4 is where real headlines get tested.

---

## Open items to verify Monday

1. Hugging Face account and token, before Sep 28
2. PyPI name: the result from T1.1 was never written into `HANDOFF.md`
3. Sun Sep 27 is also movierec's comparison report. Both projects' heaviest
   sessions fall on the same day, and again on Sun Oct 11
4. `HANDOFF.md` "Where I am" is now a snapshot. Check it stays short

---

## Recurring lessons from this week

**Every bug this week was found by running a real path.** The .gitignore
negation by `git check-ignore`. The loading break by actually calling
`load_dataset`. The config bug by the first function to resolve settings
the way the CLI will. Tests that inject dependencies directly never test
the wiring that assembles them.

**Estimates held where the source was fixed and documented.** Row counts
and nesting matched the guide exactly, unlike movierec's catalog. The
misses were the numbers that were guessed instead of looked up:
`max_length` was off by more than 3x, and the majority rate came in at
62.1% against a guess of about 60%.

**Review the fix, not only the bug report.** The first config fix
resolved the reported symptom and left the same bug one tier up.

---

## Week 2 starts here

**Mon Sep 21:** T1.4, TF-IDF + Logistic Regression. Expect it to clear the
floor. Surface vocabulary carries real signal in this corpus; T1.6
measures how much.

**Tue Sep 22:** T1.5, metrics and floor rows. **Wed Sep 23:** T1.6,
baseline writeup.

**Thu–Fri:** T3.1 RSS ingest and T3.2 SQLite store, moved ahead of the
model.

**Sun Sep 27:** T2.1 + T2.2 + T2.3, the fine-tune, about two hours. See
open item 3.