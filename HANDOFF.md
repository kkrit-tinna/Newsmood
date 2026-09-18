# Newsmood — session handoff

*Read this first, every session, before touching code. Then `CLAUDE.md` for the standing rules, and `IMPLEMENTATION_GUIDE.md` for the one task you're doing.*

---

## What this is

`newsmood` — a Python CLI that pulls financial news headlines from public RSS feeds, classifies each one positive / neutral / negative with a DistilBERT model fine-tuned on the Financial PhraseBank, aggregates them into a daily mood index, and uses the Claude API to write a short narrative summary. Output is a committed markdown report per weekday.

Successor to `social_media_sentiment_analysis`, which is archived. That project stalled on a synthetic Kaggle dataset with 191 unusable sentiment labels; the models topped out at 56%. This one uses a real corpus with three clean classes and ships as an installable tool rather than a notebook.

**Timeline:** Sep 16 – Oct 18, 2026. **Offline-first** — first working report targeted Fri Oct 2. **Budget:** 15–30 min/day Mon–Fri, 1–2 hours Sunday.

---

## Who I am

Master's student in Data Science at Northeastern, graduating Winter 2026. Currently on a data science co-op. Searching for a Spring 2027 internship, primarily targeting **Data Engineer** roles, Boston area or remote. I need visa sponsorship, so I'm applying broadly and early.

I'm comfortable with pandas, scikit-learn, and notebook-scale PyTorch. **I have not shipped a pip-installable package before**, and I have not fine-tuned a transformer to a working result — the last attempt hit 7.75%. Assume packaging, entry points, and HF Hub publishing are new to me and explain them as they come up.

**The time budget is real** and drives most scoping decisions. I am running `movie-rec-system-v2` in parallel on the same schedule, so this project gets 15–30 minutes on a weekday and 1–2 hours on Sunday — not more, because the other project needs the same again. If a task will not fit in the slot I have, say so at the *start* of the session and propose where to split it, rather than discovering it at the end.

---

## Decisions already made — do not relitigate these unless I ask

**1. A CLI package, not a web service. No cloud, no Docker, no Terraform.**
My other project (`movie-rec-system-v2`) is the AWS one. Duplicating deployment here buys no new interview material and adds a bill. Storage is SQLite. Scheduling is GitHub Actions.

**2. The model ships from the Hugging Face Hub, not inside the wheel.**
A fine-tuned DistilBERT is ~265 MB against PyPI's 100 MB file limit. Lazy download to `~/.cache/newsmood/` on first run, revision pinned in config.

**3. RSS feeds, not a news API and not Twitter.**
NewsAPI's free tier is development-use-only with a 24-hour delay and truncated content; the next tier is $449/month. Twitter free-tier read access has been closed since 2023, which kills that line of the original plan. RSS has no key, no quota, and no terms problem.

**4. One PhraseBank config only — `sentences_75agree`.**
The four agreement configs are **nested supersets**. Training on `50agree` and testing on `allagree` is testing on training data. Split within the one config, 70/15/15 stratified, seed 42. **This constraint propagates through the whole project** — if any later task evaluates across configs, every accuracy number in the repo becomes wrong.

**5. Raw text to the transformer. Cleaned text only for TF-IDF.**
The archived repo's `advanced_text_cleaning()` lemmatizes, strips punctuation, and rewrites negations as `NOT_word`. That fights WordPiece tokenization. It looks like the most reusable code in the old repo and is the piece most likely to quietly hurt the model.

**6. DistilBERT fine-tuned, not FinBERT.**
FinBERT is already trained on financial sentiment — using it means there is no fine-tuning story. FinBERT appears once, as a zero-shot reference ceiling in T2.3.

**7. Claude Haiku 4.5 writes prose; it does not classify.**
Labels come from the fine-tuned model and are passed to the API as given. Counts and the index are computed in Python. If the LLM is allowed to re-judge sentiment or add up a column, the project's central claim disappears and the arithmetic becomes unreliable.

**8. No SMOTE.** Class weights instead. Interpolating between TF-IDF vectors invents sentences nobody wrote.

**9. Offline-first. The Claude API is optional and comes last.**
`newsmood report --offline` produces the whole report, including a one-sentence template lede and the computed Why section. No API key, no paid service, no account beyond GitHub and Hugging Face — the project costs $0.00 in this mode. T4.1 would upgrade the lede sentence to a paragraph and nothing else. If it never gets built, the project is still complete.

**10. Data sources are settled: Financial PhraseBank for training, RSS for daily input. Nothing else.** Other corpora were evaluated and rejected. Don't reopen this; propose a source only if one of these two breaks.

---

## Target architecture

```
GitHub Actions (weekdays 22:00 UTC)
  └─> newsmood ingest   RSS → normalize → dedupe → SQLite
  └─> newsmood score    DistilBERT from HF Hub (cached) → label + confidence
  └─> newsmood report   aggregate → mood index → gates → Claude Haiku → markdown
  └─> commit reports/YYYY-MM-DD.md
```

**Target cost: $0.00/month in offline mode.** If online mode is enabled later, ~$0.30/month of Anthropic API, logged per call to `reports/costs.jsonl`.

---

## Where I am right now

*Last updated: Friday, Sep 18, 2026*

**Status:** T1.3 complete — VADER baseline (`feat: vader-baseline`). `baselines/vader.py` runs `vaderSentiment`'s compound score (bundled lexicon, no `nltk.download` at runtime) over raw, unmodified sentences from the pinned test split, thresholded at ±0.05 (`config/default.yaml` → `vader.positive_threshold` / `negative_threshold`, not hardcoded). `evaluation/metrics.py` has accuracy, macro-F1, and confusion-matrix functions (pure, tested without going through the CLI) — minimal set for T1.3; T1.5 extends it with weighted-F1, per-class recall, and the floor rows. `newsmood eval --model vader` prints all three. Result: **54.05% accuracy, 45.83% macro-F1 — below the 62.1% majority-class floor.** `docs/baselines.md` has the confusion matrix and three real misclassified sentences, all the same failure mode: VADER's lexicon scores "profit"/"loss" as fixed-polarity words and can't read the direction word next to them ("profit ... down from", "profit decreased", "loss narrowed"). 16 tests across `tests/test_vader.py` and `tests/test_metrics.py`, all offline (bundled lexicon, in-memory fixtures).

**Real deviations from the guide, logged in §9 and in the same commit:**
- `datasets.load_dataset("takala/financial_phrasebank", ...)` no longer works — `datasets>=4.0` dropped script-based loading and this repo never got a Parquet conversion. Fixed by downloading `FinancialPhraseBank-v1.0.zip` directly and parsing it with the original script's own logic. Same data, not a new source.
- T2.1's `max_length` placeholder (192) is superseded by the measured p95 (58) — worth remembering when T2.1 starts, 192 would pad most batches 3x+ more than needed. **But T2.1 must measure p99 before picking a final value** — p95=58 while max=150, so 64 would truncate the long tail. Decide between 96 and 128 based on the real p99.
- `pyproject.toml` never declared `pytest` anywhere despite every task's Done-when running it. Added a `dev = ["pytest"]` extra.
- `config.py`'s settings source order had the yaml file outranking explicit constructor kwargs, so any `Settings(dataset=...)` built in a test for a yaml-covered field was silently overridden back to the real yaml value. Found while wiring `predict_test_split(settings)` — it loaded the real 518-row test split instead of a tmp fixture. Fixed: `init_settings` now ranks above the yaml source.

**Schedule:** on track. `IMPLEMENTATION_GUIDE.md` §6 T3.6 (Reddit) is the first thing to drop if I fall behind; T3.4 (drift analysis) is the last, because it is the most valuable task in the project.

**Ready:**
- [x] GitHub repo `newsmood` created, public
- [x] Disk space checked, ~5 GB+ free
- [x] Repo cloned locally, venv created, `pip install -e .` working
- [ ] Hugging Face account + write token
- [ ] **Anthropic API key + $5 spend limit** — *deferred.* Claude Pro does **not** include API access; Console is a separate signup with its own prepaid balance. Offline mode means this isn't needed until T4.1 on Oct 13, if at all
- [ ] PyPI name checked (`newsmood` free or fall back to `newsmood-cli`)
- [ ] GitHub Actions secret `ANTHROPIC_API_KEY` added (Week 4)

**Known environment quirks:**
- conda `base` auto-activates in every shell. The guide assumes plain `python3 -m venv`.
- Apple Silicon: `get_device()` returns `mps`. MPS is occasionally slower than CPU for small batches — if fine-tuning looks stuck, try `--device cpu` before debugging anything else.

**Next task:** T1.4 — TF-IDF + Logistic Regression baseline. Full calendar in `IMPLEMENTATION_GUIDE.md` §0.5.

**No sessions Sunday Sep 20 or Sunday Oct 4.** Three Sundays are load-bearing and cannot move to a weekday: **Sep 27** (T2.1+T2.2+T2.3, the fine-tune), **Oct 11** (T3.4, hand-labeling in one sitting), **Oct 18** (T4.4 Actions + release). Full calendar in `IMPLEMENTATION_GUIDE.md` §0.5.

**Reference:** the archived repo is at `~/Desktop/DS Projects/Archive/Sentiment_Analysis` and on GitHub. `IMPLEMENTATION_GUIDE.md` §3 lists exactly which functions to port and which to leave behind.

---

## How I want you to work with me

- **`IMPLEMENTATION_GUIDE.md` is the source of truth. One `T#.#` task per session.** Don't skip ahead of a task's dependencies.
- **Start each session by telling me which task you're doing and whether it fits in the time I have.** If it doesn't, propose where to split it.
- If reality contradicts the spec — a config row count is off, an RSS feed is dead, a column is missing — **update the guide and log it under §9 Deviations** in the same commit. Don't work around it silently.
- **Explain the packaging and ML-ops parts as they come up.** Entry points, extras, `save_pretrained`, Hub revisions, Actions caching. I'd rather understand the thing than have it work.
- **Push back if I'm over-engineering.** The last project died of scope. If I ask for a feature that isn't in the guide, ask which task it replaces.
- **Never write a test that calls the Anthropic API.** Inject a fake client.
- **Flag anything that spends money before running it**, including a training run that will heat my laptop for 40 minutes.
- When a metric comes out worse than hoped, **write down the real number**. The drift analysis in T3.4 is the most valuable part of this project precisely because it reports a decline.
