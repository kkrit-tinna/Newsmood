"""Cleaning for the TF-IDF baseline only.

DistilBERT and VADER receive the unmodified `title`/`sentence` text — this
function must never sit in front of either. Both rely on the surface form:
WordPiece tokenization is pretrained on natural text, and VADER's lexicon
keys on capitalization and punctuation. Lemmatizing, stripping punctuation,
or rewriting negations (`NOT_word`) fights both and is a regression if
added here.
"""

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_TICKER_RE = re.compile(r"\$[A-Za-z]{1,5}\b")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_for_tfidf(text: str) -> str:
    """Lowercase, strip URLs and ticker symbols, collapse whitespace."""
    text = _URL_RE.sub(" ", text)
    text = _TICKER_RE.sub(" ", text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
