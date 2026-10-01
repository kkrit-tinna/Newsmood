# newsmood
Financial news sentiment CLI. Fine-tuned DistilBERT labels RSS headlines;
a daily markdown report is rendered from computed evidence.
`IMPLEMENTATION_GUIDE.md` is the source of truth — read the relevant
`T#.#` block before starting work. `HANDOFF.md` §Where I am says what's
done. Module layout is §3 of the guide.

## Session rules
- One `T#.#` task per session. Stop when that task's **Done when** command
  passes. Do not begin the next task.
- Do not start a task whose **Depends on** is not yet committed.
- State the task **before** writing code.
- If the real data disagrees with the guide (row count, missing config,
  dead feed), update the guide in the same commit and add today's line to
  §9 Deviations. Do not work around it silently.
- Push back on scope creep. If asked for something outside the current
  task, ask which task it replaces.

## Hard constraints
**Raw text to the transformer.** `preprocessing.py` is for TF-IDF only.
DistilBERT receives the unmodified `title`. Lemmatizing, stripping
punctuation or rewriting negations fights WordPiece tokenization.

**One PhraseBank config: `sentences_75agree`.** The four agreement configs
are nested supersets. Training on one and evaluating on a smaller one is
evaluating on training data. Never mix configs; never cross-evaluate.

**The model classifies, the LLM writes, Python computes.** Sentiment labels
come from DistilBERT and are passed to the API as given. Counts, shares and
the mood index are computed in Python and never asked of a model.

**Offline is the default path.** No module outside `reporting/claude.py`
may require an API key. `newsmood report --offline` must produce a complete
report — lede, evidence blocks, tables — with no key in the environment.

**Explanation blocks declare emptiness.** Every block in `explain.py` has a
data floor and prints an explicit "nothing to report" when unmet. A
manufactured theme reads identically to a real one.

## Module boundaries
Four rules that keep the pipeline testable. Breaking one is a refactor, not
a shortcut.
- Only `data/store.py` imports `sqlite3`. Every other module takes and
  returns plain objects.
- `cli.py` holds no business logic. If a test would need to invoke the CLI
  to exercise a rule, the rule is in the wrong module.
- `template.py` performs no arithmetic. It receives finished numbers.
  Arithmetic in a template is arithmetic nobody tests.
- `reporting/aggregate.py` owns the mood index. Anything evaluating the
  index imports that function rather than reimplementing it.

The TF-IDF vectorizer is fitted **once** in `baselines/logreg.py` on the
training split and persisted. `reporting/explain.py` loads that same
vectorizer for distinctive terms and clustering. Never fit a second one.

## Storage
One SQLite file at `data/newsmood.db`, not committed. 
Two tables:
`headlines` (one row per deduplicated headline, with its label) and `daily_index` (one row per non-empty ET day). 
The schema in `data/store.py` is the source of truth for columns.

- `model_id` records the repo@revision behind every label and index value.
  Without it, retraining silently splices two classifiers into one trend line.
- Dedupe on `title_norm`, not `url`. The same wire story appears at several
  outlets under several URLs.
- Timestamps are stored UTC via `store._ts` (fixed width). Day ranges are
  string comparisons and depend on that format.
- A day is a calendar day in America/New_York via zoneinfo. Never
  `date.today()` or a fixed UTC offset; the Actions runner's clock is UTC.

## Environment
- Python 3.11+, venv at `newsmood_env/`. Activate before running anything.
- Dependencies in `pyproject.toml`; no `requirements.txt`. Anything `newsmood score` needs is a core dependency. `[train]` holds
  training-only deps.
- Thresholds, feed URLs, model revision and band cutoffs live in `config/default.yaml`. Never hardcode one in a module.
- The model loads from the HF Hub at a **pinned revision**, cached under `~/.cache/newsmood/`. Never load from `main`.
- CLI results go to stdout; notices, progress and warnings go to stderr. Output must stay safe to pipe or redirect.

## Testing
- No test calls the Anthropic API or downloads the model. Inject fakes
  (`reporting/claude.py` takes a client; scoring takes a classifier).
- Gates, metrics and mood-index arithmetic must be tested. Those are the
  parts that fail silently and still produce a plausible report.

## Git
- Do not commit. The user reviews the diff and commits.
- Do not edit HANDOFF.md or update past §9 entries of IMPLEMENTATION_GUIDE.md unless asked; the user does close-out after review. 
  Edit IMPLEMENTATION_GUIDE.md when real data contradicts it (Session rules), and add entries in §9 when there is deviation.
- One commit per task, on `main`, message format `feat: rss-ingest` or
  `docs: model-card`.
- Never commit anything under `data/` except `data/sample/`, and never
  `models/`, `.env`, or `*.db`.
- One-line commit subjects only, e.g. `feat: rss-ingest`, `docs: model-card`.
- `reports/` **is** committed — the daily output is the visible evidence the pipeline runs.

## Scope
No web service, no cloud infrastructure, no Docker, no streaming, no ticker
extraction, no price data. Data sources are settled: Financial PhraseBank
for training, RSS for daily input. Propose another source only if one of
those two breaks.
