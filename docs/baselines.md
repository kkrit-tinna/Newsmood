# Baselines

## VADER

`vaderSentiment`'s compound score, thresholded at ±0.05 (`vader.positive_threshold`
/ `vader.negative_threshold` in `config/default.yaml`), run on the pinned
`sentences_75agree` test split (n=518). Raw sentences, unmodified — VADER
gets the same untouched text as the transformer will, not the TF-IDF cleaning
path (CLAUDE.md).

```
newsmood eval --model vader
```

| Metric | Value |
|---|---|
| Accuracy | **54.05%** |
| Macro-F1 | 45.83% |
| Weighted-F1 | 55.29% |

**Below the majority-class floor.** Always predicting "neutral" clears 62.1%
(`docs/dataset.md`) for free; VADER's lexicon-and-thresholds approach gets
54.05% — worse than guessing the most common label every time.

Per-class recall:

| Class | Recall |
|---|---|
| negative | 22.22% |
| neutral | 54.97% |
| positive | 66.92% |

Confusion matrix (rows = true, cols = predicted):

| | pred: negative | pred: neutral | pred: positive |
|---|---|---|---|
| **true: negative** | 14 | 17 | 32 |
| **true: neutral** | 19 | 177 | 126 |
| **true: positive** | 11 | 33 | 89 |

Negative headlines are the biggest casualty: only 22.22% recall, with 32 of
63 misread as positive.

### Largest single failure mode: neutral misread as positive

The largest off-diagonal cell is **true: neutral, predicted: positive — 126
of 518 rows**, more than twice the size of any other error cell. Three
examples where a single finance-specific term drives the misread:

1. **"It holds 38 percent of Outokumpu 's shares and voting rights , but in
   2001 lawmakers gave it permission to reduce the stake to 10 percent ."**
   True: neutral. Predicted: **positive** (compound: 0.1531). Driving word:
   **"shares"** — the only lexicon hit in the sentence (score 1.2). VADER's
   lexicon carries "share/shares" as inherently positive (its everyday sense
   of generosity), with no notion that here it's just the financial
   instrument. The mild negative cue ("reduce the stake") isn't in the
   lexicon at all.

2. **"Net profit in the same period in 2006 was 36.6 million euros ."**
   True: neutral. Predicted: **positive** (compound: 0.4404). Driving word:
   **"profit"** (score 1.9), the only hit. A flat factual figure with no
   comparison or judgment attached — VADER scores "profit" as a
   fixed-polarity positive token regardless of whether the sentence is
   actually making a positive claim.

3. **"The phones are targeted at first time users in growth markets ."**
   True: neutral. Predicted: **positive** (compound: 0.3818). Driving word:
   **"growth"** (score 1.6), the only hit. "Growth markets" here is a
   market-segment label (like "emerging markets"), not a claim that anything
   grew — the lexicon can't distinguish the descriptive sense from the
   directional one.

All three share the same pattern: a single financial noun carries enough
lexicon weight on its own to push a purely factual, non-evaluative sentence
past the +0.05 threshold.

### Why it fails

VADER's general-purpose lexicon scores individual words in isolation and has
no notion of financial direction. Three representative misses from the test
split:

1. **"Profit before taxes was EUR 4.0 mn , down from EUR 4.9 mn ."**
   True: negative. Predicted: **positive**. VADER's lexicon scores "profit"
   as a positive word and has no mechanism for "down from" reversing that —
   it can't see that a shrinking profit is bad news.

2. **"Finnish glass technology company Glaston Oyj Abp net profit decreased
   to 2.6 mln euro ( \$ 3.8 mln ) for the first nine months of 2007 from 7.8
   mln euro ( \$ 11.4 mln ) for the same period of 2006 ."**
   True: negative. Predicted: **positive**. Same failure mode: "profit"
   outweighs "decreased," even with the actual figures right there in the
   sentence.

3. **"The diluted loss per share narrowed to EUR 0.27 from EUR 0.86 ."**
   True: positive. Predicted: **neutral**. This is genuinely good news — a
   shrinking loss — but VADER has no entry that reads "narrowed" as an
   improvement, and "loss" alone doesn't carry enough weight past the ±0.05
   threshold to register as anything but neutral.

This is the failure the guide predicted: a lexicon built for general text has
no entries for "impairment" or "liability," and reads a word like "profit" or
"loss" as a fixed-polarity token rather than something whose meaning flips
with the number and direction next to it. This result is the motivation for
everything after it — a classifier that actually reads financial magnitude
and direction (T1.4 onward, then the fine-tuned DistilBERT) has to clear a
floor VADER can't.
