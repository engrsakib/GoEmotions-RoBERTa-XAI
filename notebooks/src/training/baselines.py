"""Classical ML baselines: TF-IDF + Logistic Regression / Linear SVM."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.label_mapping import ID2LABEL
from src.training.metrics import build_classification_report, compute_sklearn_metrics


def _build_tfidf_pipeline(classifier, max_features: int, ngram_range: tuple[int, int]) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True),
            ),
            ("clf", classifier),
        ]
    )


def train_logistic_regression(
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
) -> dict:
    pipeline = _build_tfidf_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        max_features,
        ngram_range,
    )
    return _fit_and_evaluate(pipeline, train_texts, train_labels, test_texts, test_labels, "m1_tfidf_logreg")


def train_linear_svm(
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
) -> dict:
    pipeline = _build_tfidf_pipeline(
        LinearSVC(class_weight="balanced", max_iter=2000, random_state=42),
        max_features,
        ngram_range,
    )
    return _fit_and_evaluate(pipeline, train_texts, train_labels, test_texts, test_labels, "m2_tfidf_svm")


def train_baseline(
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
) -> dict:
    """Backward-compatible alias for M1."""
    return train_logistic_regression(
        train_texts, train_labels, test_texts, test_labels, max_features, ngram_range
    )


def run_all_baselines(
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
) -> dict[str, dict]:
    return {
        "m1_tfidf_logreg": train_logistic_regression(
            train_texts, train_labels, test_texts, test_labels, max_features, ngram_range
        ),
        "m2_tfidf_svm": train_linear_svm(
            train_texts, train_labels, test_texts, test_labels, max_features, ngram_range
        ),
    }


def _fit_and_evaluate(pipeline, train_texts, train_labels, test_texts, test_labels, model_id: str) -> dict:
    pipeline.fit(train_texts, train_labels)
    preds = pipeline.predict(test_texts)
    metrics = compute_sklearn_metrics(test_labels, preds)
    return {
        "model_id": model_id,
        "pipeline": pipeline,
        "predictions": preds,
        "metrics": metrics,
        "classification_report": build_classification_report(test_labels, preds, ID2LABEL),
    }
