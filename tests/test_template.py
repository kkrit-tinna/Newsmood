"""Report template (T4.2): fixed ReportData objects in, exact markdown out.

The objects are built by hand, not by build_report_data, so these tests
exercise formatting and escaping only.
"""

from __future__ import annotations

import dataclasses
from datetime import date

from newsmood.reporting.report_data import Headline, RecommendedGroup, ReportData, SentimentSlice, SourceRow
from newsmood.reporting.template import render

DAY = date(2026, 10, 1)
MID = "owner/model@" + "a" * 40

# Fake colors, distinct from config, so a color can only have come from the slice.
RED, BLUE, GREY = "#AA0000", "#0000AA", "#777777"


def sl(sentiment, count, share):
    return SentimentSlice(sentiment, count, share, {"negative": RED, "positive": BLUE, "neutral": GREY}[sentiment])


A1 = Headline("a1", "negative", 96, False, "Regional bank shares slide on deposit outflows", "https://www.cnbc.com/a1", True, "cnbc")
A2 = Headline("a2", "negative", 57, True, "Oil slips | demand *outlook* dims", "javascript:alert(1)", False, "marketwatch")
A3 = Headline("a3", "positive", 96, False, "Chipmaker [NVDA] beats estimates", "https://example.com/a b_(c)|d", True, "cnbc")
A5 = Headline("a5", "neutral", 88, False, "Treasury auction schedule", "https://finance.yahoo.com/a5", True, "yahoo-finance")
A4 = Headline("a4", "neutral", 88, False, "Fed minutes <due> Wednesday", None, False, "unknown")


def report(**overrides) -> ReportData:
    base = ReportData(
        date=DAY,
        mood=-13.7,
        band="mildly negative",
        n_headlines=5,
        n_sources=4,
        n_unsure=1,
        n_late=0,
        low_confidence_pct=60,
        slices=(sl("negative", 2, 40.0), sl("positive", 1, 20.0), sl("neutral", 2, 40.0)),
        pie=(sl("negative", 2, 40.0), sl("neutral", 2, 40.0), sl("positive", 1, 20.0)),
        recommended=(
            RecommendedGroup("negative", 3, (A1, A2)),
            RecommendedGroup("positive", 3, (A3,)),
            RecommendedGroup("neutral", 2, (A5, A4)),
        ),
        sources=(
            SourceRow("cnbc", 2, 1, 0, 1),
            SourceRow("marketwatch", 1, 1, 0, 0),
            SourceRow("unknown", 1, 0, 1, 0),
            SourceRow("yahoo-finance", 1, 0, 1, 0),
        ),
        headlines=(A1, A2, A3, A5, A4),
        model_id=MID,
        gates="not yet implemented",
    )
    return dataclasses.replace(base, **overrides)


FOOTER = f"""---

Certainty is the model's confidence in its label; below 60% it is marked unsure.

Links go to the original publisher; some limit free articles.

Model: {MID} · Gates: not yet implemented · Not financial advice.
"""

GOLDEN = (
    """# Newsmood — 2026-10-01

**Mood index: -13.7** (mildly negative) · 5 headlines from 4 sources · model unsure on 1

## Mood at a glance

```mermaid
%%{init: {"themeVariables": {"pie1": "#AA0000", "pie2": "#777777", "pie3": "#0000AA"}}}%%
pie showData
    "Negative" : 2
    "Neutral" : 2
    "Positive" : 1
```

Negative 2 (40.0%) · Positive 1 (20.0%) · Neutral 2 (40.0%)

## What happened

_Offline report: no written summary. Every number on this page is computed from the model's labels._

## Recommended articles (highest model certainty, not most important)

### Negative

- [Regional bank shares slide on deposit outflows](<https://www.cnbc.com/a1>) · 96% · cnbc
- Oil slips \\| demand \\*outlook\\* dims · 57% (unsure) · marketwatch

_Only 2 negative headlines today._

### Positive

- [Chipmaker \\[NVDA\\] beats estimates](<https://example.com/a b_(c)\\|d>) · 96% · cnbc

_Only 1 positive headline today._

### Neutral

- [Treasury auction schedule](<https://finance.yahoo.com/a5>) · 88% · yahoo-finance
- Fed minutes \\<due\\> Wednesday · 88% · unknown

## Sources

| Source | Headlines | Negative | Neutral | Positive |
|---|--:|--:|--:|--:|
| cnbc | 2 | 1 | 0 | 1 |
| marketwatch | 1 | 1 | 0 | 0 |
| unknown | 1 | 0 | 1 | 0 |
| yahoo-finance | 1 | 0 | 1 | 0 |

<details>
<summary>All 5 headlines</summary>

| Sentiment | Certainty | Headline | Source |
|---|--:|---|---|
| negative | 96% | [Regional bank shares slide on deposit outflows](<https://www.cnbc.com/a1>) | cnbc |
| negative | 57% (unsure) | Oil slips \\| demand \\*outlook\\* dims | marketwatch |
| positive | 96% | [Chipmaker \\[NVDA\\] beats estimates](<https://example.com/a b_(c)\\|d>) | cnbc |
| neutral | 88% | [Treasury auction schedule](<https://finance.yahoo.com/a5>) | yahoo-finance |
| neutral | 88% | Fed minutes \\<due\\> Wednesday | unknown |

</details>

"""
    + FOOTER
)


def test_golden_report():
    assert render(report()) == GOLDEN


def test_render_is_deterministic():
    assert render(report()) == render(report())


def test_empty_day():
    empty = report(
        mood=None, band=None, n_headlines=0, n_sources=0, n_unsure=0,
        slices=(), pie=(), recommended=(), sources=(), headlines=(),
    )
    assert render(empty) == "# Newsmood — 2026-10-01\n\nNo headlines for this date.\n\n" + FOOTER


def test_one_headline_day():
    only = Headline("x1", "positive", 91, False, "Stocks rise", "https://example.com/x1", True, "cnbc")
    one = report(
        mood=100.0, band="strongly positive", n_headlines=1, n_sources=1, n_unsure=0,
        slices=(sl("negative", 0, 0.0), sl("positive", 1, 100.0), sl("neutral", 0, 0.0)),
        pie=(sl("positive", 1, 100.0),),
        recommended=(
            RecommendedGroup("negative", 3, ()),
            RecommendedGroup("positive", 3, (only,)),
            RecommendedGroup("neutral", 2, ()),
        ),
        sources=(SourceRow("cnbc", 1, 0, 0, 1),),
        headlines=(only,),
    )
    md = render(one)
    assert "**Mood index: +100.0** (strongly positive) · 1 headline from 1 source · model unsure on 0\n" in md
    # zero slices left out of the pie, and the lone slice keeps positive's color
    assert '```mermaid\n%%{init: {"themeVariables": {"pie1": "#0000AA"}}}%%\npie showData\n    "Positive" : 1\n```' in md
    assert "Negative 0 (0.0%) · Positive 1 (100.0%) · Neutral 0 (0.0%)" in md  # but named in the line
    assert "### Negative\n\nNo negative headlines today." in md
    assert "- [Stocks rise](<https://example.com/x1>) · 91% · cnbc\n\n_Only 1 positive headline today._" in md
    assert "### Neutral\n\nNo neutral headlines today." in md
    assert "<summary>All 1 headline</summary>" in md
    assert "| positive | 91% | [Stocks rise](<https://example.com/x1>) | cnbc |\n\n</details>" in md


def test_no_positives_day():
    no_pos = report(
        mood=-30.2, band="moderately negative", n_headlines=4, n_sources=3,
        slices=(sl("negative", 2, 50.0), sl("positive", 0, 0.0), sl("neutral", 2, 50.0)),
        pie=(sl("negative", 2, 50.0), sl("neutral", 2, 50.0)),
        recommended=(
            RecommendedGroup("negative", 3, (A1, A2)),
            RecommendedGroup("positive", 3, ()),
            RecommendedGroup("neutral", 2, (A5, A4)),
        ),
        sources=(SourceRow("cnbc", 1, 1, 0, 0), SourceRow("marketwatch", 1, 1, 0, 0), SourceRow("unknown", 1, 0, 1, 0)),
        headlines=(A1, A2, A5, A4),
    )
    md = render(no_pos)
    assert "### Positive\n\nNo positive headlines today.\n\n### Neutral" in md
    assert '"Positive"' not in md
    assert "Positive 0 (0.0%)" in md
    assert "| positive |" not in md
    assert "_Only" not in md.split("### Positive")[1].split("### Neutral")[0]


def test_no_negatives_day_keeps_each_sentiment_color():
    # Mermaid colors by position: with negative gone, neutral is slice 1 and
    # must get neutral's grey in pie1, not negative's red.
    no_neg = report(
        n_headlines=4,
        slices=(sl("negative", 0, 0.0), sl("positive", 1, 25.0), sl("neutral", 3, 75.0)),
        pie=(sl("neutral", 3, 75.0), sl("positive", 1, 25.0)),
    )
    md = render(no_neg)
    assert (
        '```mermaid\n%%{init: {"themeVariables": {"pie1": "#777777", "pie2": "#0000AA"}}}%%\n'
        'pie showData\n    "Neutral" : 3\n    "Positive" : 1\n```'
    ) in md
    assert RED not in md


def test_late_headlines_noted_under_header():
    md = render(report(n_late=2))
    assert "model unsure on 1\n\n_2 later headlines for this date arrived after its index was frozen and are not counted._\n\n## Mood" in md
    assert "_1 later headline for this date arrived after its index was frozen and is not counted._" in render(report(n_late=1))
    assert "later headline" not in render(report())


def test_title_newlines_collapse_so_rows_stay_intact():
    h = dataclasses.replace(A5, title="Two\nlines  here")
    md = render(report(headlines=(h,), recommended=(RecommendedGroup("neutral", 2, (h,)),)))
    assert "| neutral | 88% | [Two lines here](<https://finance.yahoo.com/a5>) | yahoo-finance |" in md
