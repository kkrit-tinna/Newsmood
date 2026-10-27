# Domain drift: PhraseBank to live headlines

The model was fine-tuned and tested on Financial PhraseBank (`sentences_75agree`).
This page measures it on 100 live RSS headlines I labeled by hand. Every number
below comes from `data/t34_analysis.json`, written by `scripts/t34_analyze.py`.

## Protocol

Labeling rule (fixed before sampling):
1. Label as an investor in the company, sector or market the title is about: would this
   news likely move its price up (positive), down (negative), or not at all (neutral)?
   Tone, social impact and outside knowledge don't count. If the title states the price
   reaction, that decides it.
2. Advice, how-tos, explainers, listicles and previews of scheduled events are neutral,
   unless the title states a direction ("Why is X stock rallying" is positive).
   Directional forecasts count.
3. Flag hard if still unsure after ~5 seconds, or if two labels are defensible.
   Always choose a label.

Sampling (fixed before labeling, 2026-10-12):
- Pool: every row in `data/newsmood.db` with `published_at >=` the store's first
  `fetched_at` (2026-09-25T18:39:25Z) and a model label, after `newsmood ingest` and
  `newsmood score` on Oct 12. Rows published before the first fetch (Yahoo's stale
  items) are excluded. The export exits with an error if any row in the window is unscored.
- Pool size: 168.
- Seed: 42. The pool is sorted by `id`, then `random.Random(42).sample(pool, 100)`,
  then shuffled with the same RNG.
- Sample size: 100.
- Export: `python scripts/t34_export.py` writes `data/sample/t34_blind.jsonl` (id, title,
  blank label, hard) and the model's labels to `data/t34_model_labels.jsonl`
  (gitignored, not opened until the hand labels are frozen).
- Labeling: `python scripts/t34_label.py` (titles only; never reads the model labels).

Labels frozen 2026-10-12T20:19:46-04:00, sha256 cd6d3f85fdc36341be9e401f6ee28d86f9ddca743a2fb4c7e852710a5377b50e

## Result

| Evaluation set | n | Correct | Accuracy | Wilson 95% |
|---|---|---|---|---|
| PhraseBank test (T2.3) | 518 | | **92.08%** | |
| Live headlines, all | 100 | 59 | **59.00%** | [49.20, 68.13] |
| Live, not flagged hard | 80 | 49 | 61.25% | [50.29, 71.18] |
| Live, flagged hard | 20 | 10 | 50.00% | [29.93, 70.07] |

The drop is 33.08 points. The upper bound of the live interval, 68.13%, is
still 24 points below the PhraseBank number.

### Confusion matrix

Rows are my label, columns the model's.

| | negative | neutral | positive | n |
|---|---|---|---|---|
| **negative** | 16 | 10 | 2 | 28 |
| **neutral** | 7 | 36 | 2 | 45 |
| **positive** | 5 | 15 | 7 | 27 |

| Class | Recall |
|---|---|
| negative | 16/28 = 57.14% |
| neutral | 36/45 = 80.00% |
| positive | 7/27 = 25.93% |

### By source

| Source | n | Correct | Accuracy | Wilson 95% |
|---|---|---|---|---|
| marketwatch-top | 47 | 32 | 68.09% | [53.83, 79.60] |
| cnbc-finance | 22 | 12 | 54.55% | [34.66, 73.08] |
| investing-stocks | 31 | 15 | 48.39% | [31.97, 65.16] |

The three intervals overlap; at these sizes the ordering is not established.

## Finding

Positive is the weak class: the model recovers 7 of 27 headlines I labeled
positive. The largest cell in the matrix is positive→neutral, 15 of the 41
errors, and 9 of those 15 are at confidence ≥ 0.85. The model is not unsure on
these; it is confidently reading them as neutral. They include deal and funding
news (Bumble going private, Nuvacore's raise, Amazon–Decart), launches, and
"Why is X stock rallying/gaining today?" titles, which rule 2 settled as
positive before labeling because the title states a direction.

§Six errors gives one example per off-diagonal cell.

## Six errors

Chosen by hand from the 41 disagreements. Confidence is the model's.

| # | id | Headline | Mine | Model | Conf. |
|---|---|---|---|---|---|
| 1 | `39669ea7d677` | Why is Tesla stock rallying today? | positive(flagged hard) | neutral | 0.93 |
| 2 |`88d474ea49a9` | Why is Avient stock sliding today? | negative | neutral | 0.83 |
| 3 |`98344e298195` | Why is the Chinese stock market missing the AI rally? | neutral(flagged hard) | negative | 0.79 |
| 4 |`b29aab883989` | Options traders are betting on a dramatic drop in interest rates | positive(flagged hard) | negative | 0.96 |
| 5 |`c3b7a1749901` | Fed's Kashkari says inflation is 'still too high' even after softer-than-expected PCE data, labor market is 'pretty good' | negative | positive | 0.52 |
| 6 |`eea30a1f7f88` | With mortgage rates soaring above 7.5%, the smart money rents in America | neutral | positive | 0.87 |

1. **<My prediction is positive but model says neutral. "Why is Tesla stock rallying today?">.** 'rallying' is a flag word for positive that model might miss.
2. **<My prediction is negative but model says neutral. "Why is Avient stock sliding today?">.** 'sliding' is a flag word for negative that model might miss.
3. **<My prediction is neutral but model says negative. "Why is the Chinese stock market missing the AI rally">.** 'rally' here is a noun for is a congregation, and 'missing' here is a neutral word that model might mis-categorize as negative.
4. **<My prediction is positive but model says negative. "Options traders are betting on a dramatic drop in interest rates">.** 'drop' is a flag word for negative, but here a drop in interest rate is a positive sign for market confidence: Interest rate drops primarily benefit borrowers and businesses by lowering the cost of credit, which in turn stimulates broader economic activity. 
5. **<My prediction is negative but model says positive. "Fed's Kashkari says inflation is 'still too high' even after softer-than-expected PCE data, labor market is 'pretty good'">.** This sentence has a mixed tone (leaning cautious or slightly negative on inflation, but positive on the labor market). The first half of the sentence('inflation is 'still too high') flags negative sentiment and leads the sentence, giving this headline a lean towards negative tone.
6.**<My prediction is neutral but model says positive. "With mortgage rates soaring above 7.5%, the smart money rents in America">.** 1) The model mistook 'soaring', which is a common flag word for 'positive' as positive sign, even though the entire sentence is introducing why renting is a better strategy under the higher mortgage rate (a negative sign for economy), so it should be treated as neutral.

## Limits

- **Label reliability.** A blind relabel of 20 (seed 43) agreed with my first
  labels on 14/20 = 70.00%, Cohen's κ 0.549. The reference labels are noisy,
  so the 59.00% is a measure against one labeler, not against ground truth.
- **The hard flag is a weak signal.** Of the 6 labels that changed on relabel,
  4 had not been flagged hard.
- **One story sampled twice.** `94a3bd4385dc` and `4973b35ad5f6` are the same
  Amazon–Decart story under two titles, which passed the `title_norm` dedupe.
  Both are kept; both are positive→neutral errors, so that cell counts the
  story twice.
- **Size of the drop.** The guide expected 10 to 20 points; the measured drop
  is 33.08. This is reported as a finding, not a deviation from the plan.
