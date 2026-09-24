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

*Last updated: Wednesday, Sep 23, 2026*

**Status:** T1.1–T1.6 complete. Phase 1 done. On schedule.

**Numbers later tasks depend on**

| | |
|---|---|
| Training config | `sentences_75agree`, 3,453 rows |
| Split | 70/15/15 stratified, seed 42 → 518-row test set, SHA-256 pinned in `splits_meta.json` |
| Negative share | 12.2% |
| Token length | mean 30.4 · p95 58 · max 150 |
| Majority class (floor) | 62.16% acc · 25.56% macro-F1 · 0.00% negative recall — test split, n=518. (62.1% elsewhere is the same share over the full 3,453-row corpus, not this baseline — see `docs/dataset.md`.) |
| Stratified random | 44.02% acc · 30.54% macro-F1 · 11.11% negative recall — single seeded draw, `dataset.seed=42`, not averaged over repeats |
| VADER baseline | 54.05% acc · 45.83% macro-F1 · 22.22% negative recall |
| LogReg baseline | 83.59% acc · 79.97% macro-F1 · 83.44% weighted-F1 · 77.78% negative recall |
| Human ceiling | qualitative only — up to 25% of annotators disagreed on every kept `75agree` row, no number fabricated |

**Carry forward**
- **T4.2b/T4.4** — `/models/` is gitignored, so the Actions runner won't have the fitted TF-IDF vectorizer (`models/baselines/`, persisted by `baselines/logreg.py`). Decide on Oct 5: refit in CI from the pinned split, commit the joblib file, or publish to the Hub.
- **T1.5 output** — `evaluation.metrics.calculate_comprehensive_metrics(method, y_true, y_pred, labels)` returns a plain dict keyed by method name; `evaluation/suite.py`'s `run_reference_and_baselines()` merges majority-class/stratified-random/VADER/LogReg into one table; `newsmood eval --all` writes it into `docs/baselines.md` between `<!-- eval:summary:start/end -->` markers, leaving hand-written prose below untouched. T2.3 adds `distilbert`/`finbert` rows to the same table shape; T3.5's gates read from it.
- **`perform_cross_validation()` is still unported.** §3's reuse table maps it to `evaluation/metrics.py` ("Baselines only"), but no task's Done-when requires it, and T1.6 closed without it (docs-only task). Assign it to a task in the guide or drop it from §3.
- **Lexicon facts in T1.3 prose are partly wrong.** `docs/baselines.md` §Why it fails and the guide's T1.3 block say VADER has no "liability" entry and reads "aggressive growth" as negative; measured: "liability" is −0.8 in the lexicon (compound −0.2023 alone), "aggressive growth" scores +0.25. "impairment" is correctly absent. Needs a guide edit + §9 line.
- **T1.6 output** — `docs/baselines.md` §Analysis carries 12 interpretive sentences wrapped in `<!-- CLAIM: ... -->` (invisible when rendered; `grep CLAIM:` to review). T2.3 should check DistilBERT against the four named LogReg failure types on the same test rows.
- **T2.1** — measure p99 token length before setting `max_length`. Choose 96 or 128. Not 64: it truncates the long tail, and financial sentences often carry their sentiment in the final clause.
- **T2.3** — check DistilBERT against LogReg's four named failure types on the same test rows, not just the aggregate score: object-dependent direction ("errors fell"), boilerplate outvoting direction ("second quarter of"), lone financial nouns, negation scope ("not quite cheap"). Row-level examples in docs/baselines.md §LogReg versus the floor.
- **`docs/baselines.md` VADER wording** — the line 155 CLAIM now splits negative-class errors into direction blindness (23) and missing vocabulary (15). Two phrases further down still assume one mechanism: the "Why no threshold fixes either" heading and "the class direction blindness breaks" in the sweep paragraph (line 200). Update both to match.

**Deviations so far** — full entries in `IMPLEMENTATION_GUIDE.md` §9
- `.gitignore`: `data/` → `data/*`, so `!data/sample/` can apply
- `datasets>=4.0` dropped script loading → raw zip download, SHA-256 pinned, `PhraseBankSourceChanged` on mismatch
- `max_length` placeholder 192 superseded by measured p95 of 58
- `pytest` added as a `dev` extra
- `config.py` precedence corrected to explicit > env > yaml > default, asserted in `tests/test_config.py`
- scikit-learn added as a core dependency
- `newsmood eval --all` writes only the marker-delimited summary table in `docs/baselines.md`, not the whole file — T1.3's hand-written failure-mode prose lives below it and must survive reruns

**Next task:** T3.1 — RSS ingest, Thu Sep 24 (moved up; T2.1–T2.3 on Sun Sep 27).

**Schedule:** no sessions Sunday Sep 20 or Sunday Oct 4. Three Sundays are load-bearing and cannot move to a weekday: **Sep 27** (T2.1+T2.2+T2.3, the fine-tune), **Oct 11** (T3.4, hand-labeling in one sitting), **Oct 18** (T4.4 Actions + release). Full calendar in `IMPLEMENTATION_GUIDE.md` §0.5. If I fall behind, drop T3.6 (Reddit) first and T3.4 (drift analysis) last.

**Ready**
- [x] GitHub repo `newsmood`, public
- [x] Repo cloned, `newsmood_env` created, `pip install -e ".[train]"` working
- [x] PyPI name recorded: `newsmood` or `newsmood-cli`
- [ ] Hugging Face account + write token — **needed by T2.4, Mon Sep 28**
- [ ] Anthropic API key — *deferred, optional.* Only for T4.1 online mode. Claude Pro does not include API access.

**Known environment quirks**
- Apple Silicon: `get_device()` returns `mps`. MPS is occasionally slower than CPU for small batches — if fine-tuning looks stuck, try `--device cpu` before debugging anything else.
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