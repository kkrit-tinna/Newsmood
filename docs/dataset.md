# Dataset: Financial PhraseBank

Source: [`takala/financial_phrasebank`](https://huggingface.co/datasets/takala/financial_phrasebank)
on the Hugging Face Hub. See CLAUDE.md and IMPLEMENTATION_GUIDE.md §9 for why
this is loaded by downloading `FinancialPhraseBank-v1.0.zip` directly rather
than via `datasets.load_dataset(...)`.

The raw file isn't versioned upstream — no release tags, no Parquet
conversion, and the loading script itself just points at `main`. So
`data/phrasebank.py` pins the zip's SHA-256 (mirroring the model-revision
pinning pattern for DistilBERT). Before parsing, it verifies the download
against that pin and raises `PhraseBankSourceChanged` — not a `KeyError` or
a confusing unpack error — if the file, an expected member inside it, or a
line's `sentence@label` format ever stops matching what this module was
written against.

## Config nesting

The dataset ships four configs by annotator agreement, and they are nested
supersets. Row counts, confirmed against the real download:

| Config | Rows |
|---|---|
| `sentences_allagree` | 2,264 |
| `sentences_75agree` | 3,453 |
| `sentences_66agree` | 4,217 |
| `sentences_50agree` | 4,846 |

Row counts alone don't prove nesting — matching sizes could still hide
different sentences. `nesting_report()` in `data/phrasebank.py` checks set
containment on sentence text (`allagree ⊆ 75agree ⊆ 66agree ⊆ 50agree`) and
reports any sentence that violates it. Run via:

```
python -m newsmood.data.phrasebank --check-nesting
```

Result: **holds**. No violations — every sentence in a smaller-agreement
config is present in every larger one.

**Training and evaluation use `sentences_75agree` only.** Never mix configs;
never cross-evaluate (see CLAUDE.md hard constraints).

## Label distribution — `sentences_75agree` (n=3,453)

| Label | Count | Share |
|---|---|---|
| neutral | 2,146 | 62.1% |
| positive | 887 | 25.7% |
| negative | 420 | 12.2% |

**Majority-class baseline: 62.1%** (neutral). This is the number every later
accuracy claim in this project is measured against — a classifier that
always predicts "neutral" clears 62.1% for free. Close to the guide's
estimate of "roughly 60%," recorded here as the real, measured figure.

## Split

70/15/15 stratified split of `sentences_75agree`, `random_state=42`, built by
`make_splits()` and persisted to `data/splits/` (gitignored, not committed —
regenerate with `python -m newsmood.data.phrasebank --build-splits`).
Stratification is applied on **both** peels (test off first, then train/val
off the remainder) so the minority `negative` class (12.2% of the data)
doesn't drift in the second split:

| Split | n | negative | neutral | positive |
|---|---|---|---|---|
| train | 2,417 | 12.2% | 62.1% | 25.7% |
| val | 518 | 12.2% | 62.2% | 25.7% |
| test | 518 | 12.2% | 62.2% | 25.7% |

### Split pinning

`random_state=42` alone only reproduces the split while the `datasets`
library version and its split implementation stay fixed. Since
`data/splits/` is gitignored, a library version bump could silently
regenerate a different split under the same seed and invalidate every
number measured before it. To guard against that, `save_splits()` also
writes `data/splits/splits_meta.json`, recording the config name, ratios,
seed, row count per split, the `datasets` library version used to build the
split, and a SHA-256 hash of the sorted test-set sentences.

- `load_splits()` **fails loudly** (`FileNotFoundError`) if the persisted
  splits are missing — it never silently regenerates them.
- `verify_splits()` recomputes that hash and raises `ValueError` on
  mismatch. Later tasks (baselines, fine-tuning, evaluation) call this
  before trusting any number computed against the test split.

Built with `datasets==5.0.1`.

## Token length under the DistilBERT tokenizer (`distilbert-base-uncased`)

Computed over all 3,453 sentences in `sentences_75agree`, untruncated:

| Stat | Tokens |
|---|---|
| mean | 30.4 |
| p95 | 58 |
| max | 150 |

**Note for T2.1:** the guide's placeholder starting point for
`SentimentDataset.max_length` was 192. The real p95 here is 58 — 192 would
pad most batches to more than 3x the length actually needed. T2.1 should set
`max_length` from this measured p95, not the placeholder.

## Examples

**negative:**
- "Earnings per share ( EPS ) dropped to EUR 0.21 from EUR 0.31 ."
- "ADPnews - Aug 3 , 2009 - Finnish media group Ilkka-Yhtyma Oyj HEL : ILK2S said today its net profit fell 45 % on the year to EUR 5.9 million USD 8.4 m in the first half of 2009 ."
- "Operating profit for the 12-month period decreased from EUR157 .5 m , while net sales increased from EUR634 .3 m , as compared to 2007 ."
- "ADPnews - Sep 28 , 2009 - Finnish silicon wafers maker Okmetic Oyj HEL : OKM1V said it will reduce the number of its clerical workers by 22 worldwide as a result of personnel negotiations completed today ."
- "EMSA Deputy Chairman of the Board Juri Lember told BNS on Wednesday that this was the first time he heard about the strike as the Swedish side had not informed the Estonian union yet ."

**neutral:**
- "ALEXANDRIA , Va. , Oct. 3 -- Markka A. Oksanen and Harald Kaaja , both of Helsinki , Finland , Juha Salokannel of Kangasala , Finland , and Arto Palin of Viiala , Finland , have developed a system for providing communications security ."
- "Our tools are specifically designed with the needs of both the business users and ICT experts in mind ."
- "BasWare Invoice Processing , BasWare Contract Matching , BasWare Order Matching and BasWare KPI Reporting Tool are part of the BasWare 's Enterprise Purchase to Pay solution suite ."
- "The studies are expected to start in 2008 ."
- "Philips was not available to comment on the report ."

**positive:**
- "It estimates the operating profit to further improve from the third quarter ."
- "At the end of March 2007 , the group 's order book was at EUR 39.6 mn , up 42 % from the corresponding period in 2006 ."
- "`` The lowering of prices by us and by our competitors shows that the real estate market has stabilised and returned into balance and apartments are acquiring a fair price in the eyes of our clients ."
- "2010 16 July 2010 - Finnish steel maker Rautaruukki Oyj HEL : RTRKS , or Ruukki , said today it turned to a net profit of EUR20m in the second quarter of 2010 from a net loss of EUR94m in the corresponding period last year ."
- "Operating profit improved by 44.0 % to ER 4.7 mn from EUR 3.3 mn in 2004 ."

## Demo sample

`data/sample/headlines_200.jsonl` — 200 sentences drawn from the **test
split** (`shuffle(seed=42).select(range(200))`), committed for `make demo`.
Drawn from test rather than a separate carve-out before splitting, since the
corpus is small (3,453 rows) and the sample isn't used for any accuracy
claim — it only needs to be held out of training, which the test split
already guarantees.
