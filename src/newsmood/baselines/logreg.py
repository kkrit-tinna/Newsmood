"""TF-IDF + Logistic Regression baseline.

Preprocessed text only (`preprocessing.py`) — unlike VADER and DistilBERT,
which read the raw sentence. The vectorizer is fit on the training split
only and never re-fit; `transform` (never `fit_transform`) is the only
thing test data touches. Vectorizer and classifier stay as two separate
objects, not one `Pipeline`, because `reporting/explain.py` (T4.2b) reuses
the fitted vectorizer by itself.
"""

from __future__ import annotations

import json
import joblib
from pathlib import Path

import sklearn
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from newsmood.config import Settings
from newsmood.data.phrasebank import LABEL_NAMES, load_splits, verify_splits
from newsmood.preprocessing import clean_for_tfidf


def fit(sentences: list[str], labels: list[str], settings: Settings) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Fit the vectorizer and the classifier on the training split only.

    Returns them as two separate objects rather than a `Pipeline` — the
    vectorizer alone is what `reporting/explain.py` needs later.
    """
    cleaned = [clean_for_tfidf(s) for s in sentences]

    vectorizer = TfidfVectorizer(
        ngram_range=tuple(settings.logreg.ngram_range),
        min_df=settings.logreg.min_df,
        max_df=settings.logreg.max_df,
        max_features=settings.logreg.max_features,
    )
    X_train = vectorizer.fit_transform(cleaned)

    classifier = LogisticRegression(
        class_weight=settings.logreg.class_weight,
        max_iter=settings.logreg.max_iter,
        random_state=settings.dataset.seed,
    )
    classifier.fit(X_train, labels)

    return vectorizer, classifier


def predict(vectorizer: TfidfVectorizer, classifier: LogisticRegression, sentences: list[str]) -> list[str]:
    """Score sentences with an already-fit vectorizer/classifier. Never fits."""
    cleaned = [clean_for_tfidf(s) for s in sentences]
    X = vectorizer.transform(cleaned)
    return list(classifier.predict(X))


def _test_sentence_sha256(splits_dir: Path) -> str:
    with open(splits_dir / "splits_meta.json") as f:
        return json.load(f)["test_sentence_sha256"]


def save_artifacts(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    settings: Settings,
    train_row_count: int,
) -> None:
    """Persist the vectorizer and classifier as separate joblib files, plus
    a meta.json tying them to the exact pinned split they were fit from.

    The hash lets a later loader detect that the splits have moved on since
    these artifacts were fit; the sklearn version is recorded because a
    pickle can fail to load across scikit-learn versions."""
    artifact_dir = Path(settings.logreg.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, artifact_dir / "tfidf_vectorizer.joblib")
    joblib.dump(classifier, artifact_dir / "logreg_classifier.joblib")

    meta = {
        "sklearn_version": sklearn.__version__,
        "train_row_count": train_row_count,
        "test_sentence_sha256": _test_sentence_sha256(Path(settings.dataset.splits_dir)),
    }
    with open(artifact_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_artifacts(settings: Settings) -> tuple[TfidfVectorizer, LogisticRegression]:
    artifact_dir = Path(settings.logreg.artifact_dir)
    vectorizer = joblib.load(artifact_dir / "tfidf_vectorizer.joblib")
    classifier = joblib.load(artifact_dir / "logreg_classifier.joblib")
    return vectorizer, classifier


def load_vectorizer(settings: Settings) -> TfidfVectorizer:
    """Load the persisted TF-IDF vectorizer alone.

    This is the only vectorizer the project should ever fit —
    `reporting/explain.py` (T4.2b) calls this for distinctive terms and
    clustering rather than fitting a second one."""
    vectorizer, _ = load_artifacts(settings)
    return vectorizer


def predict_test_split(settings: Settings) -> tuple[list[str], list[str], list[str]]:
    """Fit on the pinned train split, score the pinned test split.

    Returns (sentences, y_true, y_pred) so callers can both score and
    inspect individual misclassifications. Persists the fitted vectorizer
    and classifier as a side effect, so later callers (T4.2b) load the
    same vectorizer instead of fitting a second one.
    """
    splits_dir = Path(settings.dataset.splits_dir)
    splits = load_splits(splits_dir)
    verify_splits(splits_dir)

    train: Dataset = splits["train"]
    test: Dataset = splits["test"]

    vectorizer, classifier = fit(
        list(train["sentence"]),
        [LABEL_NAMES[i] for i in train["label"]],
        settings,
    )
    save_artifacts(vectorizer, classifier, settings, train_row_count=len(train))

    sentences = list(test["sentence"])
    y_true = [LABEL_NAMES[i] for i in test["label"]]
    y_pred = predict(vectorizer, classifier, sentences)
    return sentences, y_true, y_pred
