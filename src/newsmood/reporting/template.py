"""Markdown for one day's report (T4.2). Formats a finished ReportData and
nothing else: no arithmetic, no ordering, no I/O. Every number arrives
computed by reporting/report_data.py.

Escaping happens here. Titles and sources are feed text, so markdown and
HTML-significant characters are backslash-escaped, and a url sits in the
<...> link form so spaces and parentheses can't end it early. Both escape `|`,
since every title and url may land in a table cell.
"""

from __future__ import annotations

from newsmood.reporting.explain import Cluster, Explanation
from newsmood.reporting.report_data import Headline, RecommendedGroup, ReportData

OFFLINE_PLACEHOLDER = "_Offline report: no written summary. Every number on this page is computed from the model's labels._"
PUBLISHER_NOTE = "Links go to the original publisher; some limit free articles."
MOVEMENT_NOTE = (
    "\"Of the day's movement\" is a headline's certainty over the sum of every non-neutral headline's certainty: "
    "its part in moving the index. Neutral headlines move it by 0."
)
NO_THEME = "no clear theme today"
RECOMMENDED_HEADING = "## Recommended articles (highest model certainty, not most important)"

_TEXT_SPECIAL = set("\\`*_[]<>|~#")
_URL_SPECIAL = set("\\<>|")


def render(data: ReportData, explanation: Explanation | None = None) -> str:
    """The full report as markdown, ending in one newline. Without an
    explanation (T4.2b), "What happened" keeps the offline placeholder."""
    parts = [f"# Newsmood — {data.date.isoformat()}"]
    if data.is_empty:
        parts.append("No headlines for this date.")
    else:
        parts += [
            _header(data),
            _glance(data),
            _what_happened(explanation),
            _recommended(data),
            _sources(data),
            _all_headlines(data),
        ]
    parts.append(_footer(data))
    return "\n\n".join(parts) + "\n"


def _header(data: ReportData) -> str:
    line = (
        f"**Mood index: {data.mood:+.1f}** ({data.band}) · "
        f"{_plural(data.n_headlines, 'headline')} from {_plural(data.n_sources, 'source')} · "
        f"model unsure on {data.n_unsure}"
    )
    if data.n_late:
        line += (
            f"\n\n_{_plural(data.n_late, 'later headline')} for this date arrived after its index was "
            f"frozen and {'is' if data.n_late == 1 else 'are'} not counted._"
        )
    return line


def _glance(data: ReportData) -> str:
    # Mermaid assigns pie1, pie2, ... by slice position, so the theme lists
    # exactly the slices drawn, in the order drawn (data.pie). The line under
    # the pie always names all three.
    theme = ", ".join(f'"pie{i}": "{s.color}"' for i, s in enumerate(data.pie, 1))
    pie = "\n".join(f'    "{s.sentiment.capitalize()}" : {s.count}' for s in data.pie)
    line = " · ".join(f"{s.sentiment.capitalize()} {s.count} ({s.share_pct:.1f}%)" for s in data.slices)
    return (
        "## Mood at a glance\n\n```mermaid\n"
        f'%%{{init: {{"themeVariables": {{{theme}}}}}}}%%\npie showData\n{pie}\n```\n\n{line}'
    )


def _what_happened(explanation: Explanation | None) -> str:
    if explanation is None:
        return "## What happened\n\n" + OFFLINE_PLACEHOLDER
    parts = ["## What happened", _text(explanation.lede), _clusters(explanation)]
    if explanation.oov is not None:
        parts.append(_oov(explanation))
    if explanation.fit is not None:
        parts.append(_fit(explanation))
    return "\n\n".join(parts)


def _clusters(e: Explanation) -> str:
    if not e.clustered:
        return f"**Clusters:** {NO_THEME} ({_plural(e.n_headlines, 'headline')}; clustering needs {e.min_headlines})."
    if not e.clusters:
        return f"**Clusters:** {NO_THEME}."
    return "**Clusters:**\n\n" + "\n".join(_cluster(c) for c in e.clusters)


def _cluster(c: Cluster) -> str:
    split = ", ".join(f"{n} {s}" for s, n in (("negative", c.n_negative), ("positive", c.n_positive), ("neutral", c.n_neutral)) if n)
    return (
        f"- {_plural(c.size, 'headline')} on {', '.join(_text(t) for t in c.name)}: {split} · "
        f"{c.share_pct:+.1f}% of the day's movement"
    )


def _oov(e: Explanation) -> str:
    o = e.oov
    return (
        f"_Themes use the PhraseBank baseline's vocabulary: {o.rate_pct:.1f}% of this day's words "
        f"({o.n_oov} of {o.n_tokens}) are outside it._"
    )


def _fit(e: Explanation) -> str:
    f = e.fit
    return f"_Themes use TF-IDF fit on {_plural(f.n_headlines, 'headline')} ({f.first.isoformat()}–{f.last.isoformat()})._"


def _recommended(data: ReportData) -> str:
    return "\n\n".join([RECOMMENDED_HEADING] + [_group(g) for g in data.recommended])


def _group(g: RecommendedGroup) -> str:
    head = f"### {g.sentiment.capitalize()}"
    if not g.items:
        return f"{head}\n\nNo {g.sentiment} headlines today."
    items = "\n".join(f"- {_title(h)} · {_certainty(h)}{_contribution(h)} · {_text(h.source)}" for h in g.items)
    if g.shortfall:
        items += f"\n\n_Only {_plural(len(g.items), g.sentiment + ' headline')} today._"
    return f"{head}\n\n{items}"


def _sources(data: ReportData) -> str:
    rows = "\n".join(
        f"| {_text(s.source)} | {s.total} | {s.negative} | {s.neutral} | {s.positive} |" for s in data.sources
    )
    return (
        "## Sources\n\n"
        "| Source | Headlines | Negative | Neutral | Positive |\n"
        "|---|--:|--:|--:|--:|\n" + rows
    )


def _all_headlines(data: ReportData) -> str:
    rows = "\n".join(
        f"| {h.sentiment} | {_certainty(h)} | {_title(h)} | {_text(h.source)} |" for h in data.headlines
    )
    return (
        f"<details>\n<summary>All {_plural(data.n_headlines, 'headline')}</summary>\n\n"
        "| Sentiment | Certainty | Headline | Source |\n"
        "|---|--:|---|---|\n" + rows + "\n\n</details>"
    )


def _footer(data: ReportData) -> str:
    return (
        "---\n\n"
        f"Certainty is the model's confidence in its label; below {data.low_confidence_pct}% it is marked unsure.\n\n"
        f"{MOVEMENT_NOTE}\n\n"
        f"{PUBLISHER_NOTE}\n\n"
        f"Model: {data.model_id} · Gates: {data.gates} · Not financial advice."
    )


def _title(h: Headline) -> str:
    text = _text(h.title)
    return f"[{text}](<{_url(h.url)}>)" if h.linked else text


def _certainty(h: Headline) -> str:
    return f"{h.certainty_pct}% (unsure)" if h.unsure else f"{h.certainty_pct}%"


def _contribution(h: Headline) -> str:
    return "" if h.contribution_pct is None else f" · {h.contribution_pct:.1f}% of the day's movement"


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _text(s: str) -> str:
    s = " ".join(s.split())  # a newline would end a list item or table row
    return "".join("\\" + c if c in _TEXT_SPECIAL else c for c in s)


def _url(u: str) -> str:
    return "".join("\\" + c if c in _URL_SPECIAL else c for c in u)
