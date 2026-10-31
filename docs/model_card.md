# Model card — `newsmood-distilbert-financial`

DistilBERT (`distilbert-base-uncased`) fine-tuned to label financial news headlines **negative**, **neutral** or **positive**. Published as [`kkrit-tinna/newsmood-distilbert-financial`](https://huggingface.co/kkrit-tinna/newsmood-distilbert-financial); `newsmood` loads it at a pinned revision.

**Not financial advice.** The model labels the sentiment of text. It does not predict prices or returns.

## Intended use

- Labeling English financial headlines in aggregate, as the input to `newsmood`'s daily mood index.
- Studying how a model trained on a curated corpus behaves on live news.

## Not intended for

- Trading, investment or credit decisions, or any automated action on a single label.
- Predicting prices or returns.
- Non-English text, or text outside financial news.
- Treating confidence as a probability of being right. On live headlines, 9 of the 15 positive→neutral errors were at confidence ≥ 0.85 (`docs/drift.md`).

## Training data and provenance

[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank), `sentences_75agree` config only: 3,453 analyst-style sentences from a Finnish corpus assembled in 2014, kept where at least 75% of annotators agreed on the label. The dataset is licensed CC BY-NC-SA 3.0. The source zip is pinned by SHA-256, so a changed upstream file fails loudly (`docs/dataset.md`).

## Split policy

70/15/15, stratified, seed 42; 2,417 training rows and 518 test rows. The persisted splits are never silently regenerated, and a SHA-256 of the test sentences is checked before any number is trusted.

**One config only.** The four PhraseBank agreement configs are nested supersets. Training on one and evaluating on a smaller one is evaluating on training data, so this model is trained, validated and tested inside `sentences_75agree`, and never cross-evaluated.

## Training setup

- Epoch 3 of 3 is shipped. It had the lowest validation loss, 0.1941.
- EarlyStopping never fired. At the configured patience it could not have in a 3-epoch run.
- Epoch 2's 94.02% validation accuracy is not the published number. The published number is the test accuracy below.
- Loss weighted by class on the training split: negative ×2.74, neutral ×0.54, positive ×1.30.
- The transformer receives raw text; the cleaning in `preprocessing.py` is for the TF-IDF baseline only. `max_length` 96 with dynamic padding.
- Trained on MPS. A retrain is reproducible in substance; it is bit-exact only with `newsmood train --device cpu`, and a CPU run is not expected to match the published MPS-trained weights.

Remaining hyperparameters are under `training:` in `src/newsmood/default.yaml`.

## Test metrics

Pinned test split, n=518 (`docs/baselines.md`).

| Method | Accuracy | Macro-F1 | Recall (neg / neu / pos) |
|---|---|---|---|
| Majority class (floor) | 62.16% | 25.56% | 0.00% / 100.00% / 0.00% |
| VADER | 54.05% | 45.83% | 22.22% / 54.97% / 66.92% |
| Logistic Regression (TF-IDF) | 83.59% | 79.97% | 77.78% / 90.06% / 70.68% |
| **This model** | **92.08%** | **90.19%** | **96.83% / 93.17% / 87.22%** |
| FinBERT (trained on PhraseBank, reference only) | 95.37% | 94.45% | 98.41% / 94.10% / 96.99% |

**FinBERT is a reference, not a ceiling.** `ProsusAI/finbert` was itself trained on PhraseBank, so it is not zero-shot here: it has probably seen many of these test sentences, and the overlap can't be measured. Its number is contaminated, and the gap to this model does not measure headroom.

The model makes 41 errors, against LogReg's 85. The largest confusion is neutral→positive (17 of the 41). Negative has the highest recall but the lowest precision, 81.33%.

## Live-headline result

On 100 live RSS headlines labeled by hand, the model scores **59.00%** (59/100, Wilson 95% [49.20, 68.13]), against 92.08% on the PhraseBank test split. Positive is the weak class: recall **25.93%** (7/27), against negative 57.14% and neutral 80.00%.

Treat the live number, not the test number, as the guide to behavior on real headlines. It is measured against one labeler: a blind relabel of 20 agreed on 14/20 = 70.00% (Cohen's κ 0.549). Protocol and error examples are in `docs/drift.md`.

## Limitations

- **Domain drift.** Trained on analyst sentences, run on live headlines: a 33.08-point drop, with positive recall at 25.93% (`docs/drift.md`).
- **Near-duplicates pass the dedupe.** `title_norm` catches titles that are identical after normalization. One story under two titles is stored and counted twice; the live sample drew the Amazon–Decart story this way.
- **Dedupe drops the syndication count.** A wire story at several outlets under one title is stored once, and the number of outlets is not kept, so the index cannot weight a story by its spread.
- **English only.**
- **2014 corpus.** The training vocabulary and events are old relative to the headlines it scores.
- **Not yet done:** comparing DistilBERT's errors with LogReg's failure types on the same rows.
- **Untested hypothesis:** the class weights may be why errors lean negative (`docs/baselines.md`). Confirming it needs a run without them.

## Speed

Measured once on the machine below; expect different numbers on other hardware. Run the command to measure your own.

Reproduce with:

```
newsmood score --file data/sample/headlines_200.jsonl --benchmark
```

| | |
|---|---|
| Hardware | Apple M3 Pro, CPU only (no GPU/MPS) |
| Software | torch 2.14.0, 5 threads |
| Settings | batch size 32, dynamic padding, `max_length` 96 |
| Input | 200 rows, 7 batches, one untimed warmup batch |
| Throughput | **164.9 headlines/s** (1.21 s for 200) |
| Batched latency, per headline | p50 6.09 ms · p95 6.72 ms |
| Cold latency | 3.05 s (median of 3, range 3.03–3.05 s) |

At this rate, 1,000 headlines take about 6 s on an M3 Pro CPU.

**Batched latency** is each batch's wall time divided by the number of rows in it. The percentiles rest on only 7 batches. The final batch has 8 rows instead of 32, so its fixed per-batch cost is shared by fewer headlines, and it probably sets the p95.

**Cold latency** is the wall time of a fresh `newsmood score --text` process with the model already in `~/.cache/newsmood/`. It includes interpreter start, importing torch and transformers, and loading the model. It is what a user waits for when they run the command once. Inference itself takes milliseconds of that time.

**Input caveat.** The 200 rows are held-out Financial PhraseBank sentences (mean about 30 tokens), not live headlines. Live RSS headlines are usually shorter, and with dynamic padding shorter batches cost less, so live throughput is likely higher, not lower.