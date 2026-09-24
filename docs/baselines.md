# Baselines

<!-- eval:summary:start -->
All rows below are measured on the pinned test split (n=518).

| Method | Accuracy | Macro-F1 | Weighted-F1 | Recall (negative / neutral / positive) |
|---|---|---|---|---|
| Majority class | 62.16% | 25.56% | 47.66% | 0.00% / 100.00% / 0.00% |
| Stratified random | 44.02% | 30.54% | 43.99% | 11.11% / 60.25% / 20.30% |
| VADER | 54.05% | 45.83% | 55.29% | 22.22% / 54.97% / 66.92% |
| Logistic Regression | 83.59% | 79.97% | 83.44% | 77.78% / 90.06% / 70.68% |
| Human ceiling | — | — | — | Not a number to compare against directly: `sentences_75agree` keeps only rows where at least 75% of annotators agreed, so up to 25% of annotators disagreed with the kept label on every one of them. A model scoring in the high-80s/low-90s may be brushing a ceiling inherent to the labels, not still leaving headroom on the table. |

"Stratified random" is a single seeded draw from the training label distribution (`dataset.seed=42`), not an average over repeats — rerun with the same seed to reproduce it exactly.
<!-- eval:summary:end -->

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

**Below the majority-class floor.** Always predicting "neutral" clears
62.16% on this same test split (Majority class row, table above) for free;
VADER's lexicon-and-thresholds approach gets 54.05% — worse than guessing
the most common label every time. (`docs/dataset.md`'s 62.1% is the same
share measured over the full 3,453-row corpus rather than the 518-row test
split — close, not identical, because the split isn't exactly proportional
at that rounding.)

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

## Analysis

Every number in this section was recomputed from the pinned test split
(n=518) on 2026-09-23 — `newsmood eval --all`, plus direct lookups against
`SentimentIntensityAnalyzer().lexicon` and the persisted LogReg artifacts in
`models/baselines/`. Confusion-cell counts are out of 518.

### Why VADER fails on financial text

VADER gets 238 of 518 test sentences wrong. The errors fall into two failure
modes that point in opposite directions: one misses sentiment that is present,
the other reports sentiment that is absent.

**1. Direction blindness.** VADER's lexicon has no entry for "down", "fell",
"decreased", "narrowed", "rose" or "up". The one direction verb it does carry,
"increased" (+1.1), has no counterpart — "decreased" is absent.

| Sentence | True | VADER | Compound | Lexicon hits |
|---|---|---|---|---|
| "Profit before taxes was EUR 4.0 mn , down from EUR 4.9 mn ." | negative | positive | 0.4404 | profit +1.9 |
| "Finnish glass technology company Glaston Oyj Abp net profit decreased to 2.6 mln euro … from 7.8 mln euro …" | negative | positive | 0.4404 | profit +1.9 |
| "The diluted loss per share narrowed to EUR 0.27 from EUR 0.86 ." | positive | neutral | −0.0258 | loss −1.3, share +1.2 |

The first two score exactly the same compound, 0.4404, as the neutral "Net
profit in the same period in 2006 was 36.6 million euros ." 32 of the 63
true-negative sentences are predicted positive, and 33 of 63 have a compound
score above zero. Negative recall is 22.22%, VADER's lowest class.

**CLAIM: Negative is VADER's worst class for two independent reasons that need different fixes. In 23 sentences a positive financial noun sits beside a direction word the lexicon doesn't carry, and VADER predicted positive in all 23 without exception — the noun is scored, the reversal is invisible. In another 15 the sentence contains no lexicon entry at all, so the compound is exactly 0 and the ±0.05 band forces neutral; VADER predicted neutral for all 15. The first is a compositional failure: the words are present and their combination is misread. The second is a coverage failure: the vocabulary of financial decline is absent from a general-English lexicon. Of the 49 negative errors, 23 are the first kind and 15 the second, with 9 more predicted positive without a listed direction word and 2 whose hits nearly cancelled inside the band.**

**2. Single-noun false positives.** VADER's largest confusion cell is true
neutral → predicted positive: 126 of 518. 126 of the 322 true-neutral
sentences (39.1%) score at or above +0.05.

| Sentence | True | VADER | Compound | Only lexicon hit |
|---|---|---|---|---|
| "It holds 38 percent of Outokumpu 's shares and voting rights , but in 2001 lawmakers gave it permission to reduce the stake to 10 percent ." | neutral | positive | 0.1531 | shares +1.2 |
| "Net profit in the same period in 2006 was 36.6 million euros ." | neutral | positive | 0.4404 | profit +1.9 |
| "The phones are targeted at first time users in growth markets ." | neutral | positive | 0.3818 | growth +1.6 |

**CLAIM: In financial text "shares", "profit" and "growth" are the subject matter, not an evaluation of it. A general-English lexicon assigns them fixed positive polarity, so VADER reads the topic of a sentence as its sentiment: 126 of the 322 true-neutral sentences, 39.1%, score at or above +0.05 on a finance noun alone.**

**Why no threshold fixes either.** 44 test sentences score exactly 0.4404.
Their true labels are 18 negative, 10 neutral and 16 positive. A threshold maps
each compound value to one label, so at least 26 of these 44 are wrong under
every possible threshold. All 18 negatives at 0.4404 mention profit, 7 of the
10 neutrals mention profit, and the positives include "Operating profit rose to
EUR 3.11 mn from EUR 1.22 mn …". 221 sentences score exactly 0.0 — 177 neutral,
29 positive, 15 negative. Whichever label a threshold gives 0.0, at least 44 of
the 221 are wrong: labelling it positive or negative recovers 29 or 15
sentences at the cost of 177 neutrals.

Sweeping the thresholds confirms it. The grid set the positive and negative
cutoffs independently, each from −0.95 to +0.95 in 0.05 steps, skipping pairs
where the positive cutoff is below the negative one (780 pairs remain), and
tuned them on the test split itself, which flatters VADER.
`scripts/vader_threshold_sweep.py` reproduces every row below.

| Setting | Positive / negative cutoff | Accuracy | Macro-F1 | Negative recall |
|---|---|---|---|---|
| Configured | +0.05 / −0.05 | 54.05% | 45.83% | 22.22% |
| Best macro-F1 on grid | +0.40 / −0.25 | 61.00% | 48.07% | 20.63% |
| Best macro-F1 above floor | +0.50 / −0.25 | 62.74% | 46.11% | 20.63% |
| Best accuracy on grid | +0.70 / −0.30 | 64.48% | 42.73% | 19.05% |
| Symmetric ±0.60 | +0.60 / −0.60 | 61.78% | 33.71% | 0.00% |

64.48% is a three-way tie at +0.70 with the negative cutoff at −0.30, −0.40 or
−0.45; the row shows the best macro-F1 of the three.

The best macro-F1 threshold still lands below the 62.16% floor on accuracy.

Threshold tuning trades one metric for another without recovering the class it
fails on. Not one pair with a negative cutoff below zero raises negative recall
above the configured 22.22% — the class direction blindness breaks is the class
no usable cutoff can recover; at the extreme, symmetric ±0.60 recovers none of
it, 0.00% negative recall at 61.78% accuracy. The 276 pairs that do exceed it
set the negative cutoff at zero or above, labelling all 221 zero-score
sentences negative, and none exceeds 31.66% accuracy, which is below even the
44.02% stratified-random baseline. Of the 780 pairs, 102 clear the floor on
accuracy, and the best of those gains 0.28 points of macro-F1 over the
configured setting. Above about 63.5% accuracy the trade-off is strictly
negative: the best-accuracy settings sit at 40–43% macro-F1, below the
configured 45.83%, because accuracy there is bought by predicting neutral
88–92% of the time, against 43.82% at the configured cutoffs.

**CLAIM: No threshold can fix a scorer that gives a profit fall, a flat profit report and a profit rise the same number; the fix has to change what gets scored (the direction word, and what it modifies), not where the cut falls.**

### LogReg versus the floor

LogReg (TF-IDF over 1–3-grams, `min_df=2`, 11,169 features, fitted on the
2,417-row training split) reaches 83.59% accuracy — 21.43 points above the
majority-class floor of 62.16% — and 79.97% macro-F1 against the floor's
25.56%. Per-class recall is 77.78% negative (49/63), 90.06% neutral (290/322)
and 70.68% positive (94/133). It makes 85 errors, and its largest confusion
cell is true positive → predicted neutral, 33 of 518.

The largest learned weights are direction words, the same words VADER's
lexicon lacks. Toward negative: "down" 3.21, "decreased" 2.77, "fell" 1.96.
Toward positive: "rose" 2.50, "increase" 2.30, "increased" 2.07, "up" 1.84.
LogReg classifies all three of VADER's direction-blind sentences correctly,
and two of the three single-noun sentences.

**CLAIM: LogReg clears the floor by 21.43 points using nothing but n-gram counts, and the largest learned weights are precisely the direction words VADER's lexicon lacks: "down" 3.21, "decreased" 2.77, "fell" 1.96 toward negative; "rose" 2.50, "increase" 2.30, "increased" 2.07, "up" 1.84 toward positive. It classifies all three of VADER's direction-blind sentences correctly. What this shows is that the direction vocabulary is learnable from 2,417 labelled sentences without any model of syntax. It does not show how much of the 21 points those words account for — that would need an ablation, stripping direction terms from the feature set and remeasuring. Worth running only if the transformer's margin over LogReg turns out to be small.**

**CLAIM: That is partly a property of the corpus: PhraseBank sentences are short, single-clause, period-over-period reporting, the easiest possible case for a bag of n-grams, and RSS headlines should not be assumed to be as easy.**

Remaining errors, from the 85:

- **Direction applied to the wrong object.** "The number of collection errors
  fell considerably , and operations speeded up ." True: positive. Predicted:
  negative. "fell" is the strongest feature pushing toward negative, and
  "errors" is not in the vocabulary.
- **Boilerplate outvoting the direction word.** "Operating profit fell from
  EUR 7.9 mn in the second quarter of 2005 to EUR 5.1 mn in the second quarter
  of 2006 ." True: negative. Predicted: positive (logit 0.98 vs 0.76). "fell"
  and "profit fell" push toward negative; "second quarter of", "the second" and
  "quarter of" push toward positive and win. The Teleste sentence ("net profit
  decreased to EUR 5.5 million … from EUR 9.4 million") fails the same way,
  with "eur million" as the top feature pushing toward positive.
- **The single-noun false positive, inherited.** "Net profit in the same period
  in 2006 was 36.6 million euros ." True: neutral. Predicted: positive — the
  same sentence VADER misses.
- **Negation.** "Investors will continue being interested in the company 's
  share although it is not quite cheap" True: positive. Predicted: neutral.
  "not" is a single feature weighted 0.98 toward neutral, and "not quite" is
  not in the vocabulary.

**CLAIM: A bag of n-grams adds up independent feature weights, so it cannot represent that a direction word's polarity depends on its object ("profit fell" is bad, "errors fell" is good) except by memorising each object–direction pair as its own n-gram, and `min_df=2` discards any pair seen fewer than twice in training.**

**CLAIM: The representation has no notion of which tokens matter: reporting boilerplate like "second quarter of" carries class weight from the training sentences it co-occurred with, and in a long sentence those incidental n-grams can outvote the one word that carries the direction.**

<!-- CLAIM: Negation has no scope in this representation — "not" is a feature in its own right and cannot flip the polarity of the word it modifies. -->

### Why both F1 columns are reported

Majority class scores 62.16% accuracy and 25.56% macro-F1. Its neutral F1 is
2 × 0.6216 / (1 + 0.6216) = 76.67%, and its negative and positive F1 are both
0. So macro-F1 is 76.67% / 3 = 25.56%, and weighted-F1 is
0.6216 × 76.67% = 47.66%.

Stratified random scores 44.02% accuracy and 30.54% macro-F1. It draws
predictions independently from the training distribution (62.14% neutral,
25.69% positive, 12.16% negative), so a prediction matches the label when two
independent draws coincide. In expectation that is Σ p_train × p_test =
46.71%, and this seeded draw landed at 44.02%. Always predicting neutral
matches 62.16% of the time, and for a class share p, p² < p, so random
sampling falls below the floor on accuracy. On macro-F1 the ranking reverses.
When prediction and label distributions match, each class's expected F1
roughly equals its share, giving an expected macro-F1 of about 33.33%. The
seeded draw scored 30.54%, above the majority class's 25.56%.

VADER shows the same split: 54.05% accuracy is below the floor, and 45.83%
macro-F1 is well above it.

**CLAIM: Accuracy alone rewards agreeing with the 62% neutral majority and macro-F1 alone hides how often the dominant class is right; only with both columns does the table show that VADER is simultaneously worse than doing nothing and better than doing nothing, depending on whether the minority classes count.**

**CLAIM: For this project macro-F1 is the column that matters more: the T3.3 mood index scores neutral as 0 and is driven entirely by positive and negative labels, so a classifier with the floor's 0.00% negative and positive recall would report a mood of exactly 0 every day regardless of the news.**
