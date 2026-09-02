"""TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.label_mapping import ID2LABEL
from src.training.metrics import build_classification_report, compute_sklearn_metrics


def train_baseline(
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
) -> dict:
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True),
            ),
            (
                "clf",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            ),
        ]
    )

    pipeline.fit(train_texts, train_labels)
    preds = pipeline.predict(test_texts)
    metrics = compute_sklearn_metrics(test_labels, preds)

    return {
        "pipeline": pipeline,
        "predictions": preds,
        "metrics": metrics,
        "classification_report": build_classification_report(test_labels, preds, ID2LABEL),
    }
