"""Registry of eight recommended models/algorithms for GoEmotions classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelFamily = Literal["baseline", "transformer", "xai"]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    family: ModelFamily
    description: str
    huggingface_id: str | None = None
    use_focal_loss: bool = False
    recommended: bool = False


# Eight models ranked for this 7-class imbalanced GoEmotions task
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "m1_tfidf_logreg": ModelSpec(
        id="m1_tfidf_logreg",
        name="TF-IDF + Logistic Regression",
        family="baseline",
        description="Fast classical baseline; macro-F1 sanity floor (~0.50).",
    ),
    "m2_tfidf_svm": ModelSpec(
        id="m2_tfidf_svm",
        name="TF-IDF + Linear SVM",
        family="baseline",
        description="Strong linear separator with balanced class weights.",
    ),
    "m3_roberta_base": ModelSpec(
        id="m3_roberta_base",
        name="RoBERTa-base Fine-Tuning",
        family="transformer",
        huggingface_id="roberta-base",
        description="Primary production encoder; matches packages/model service.",
        recommended=True,
    ),
    "m4_roberta_focal": ModelSpec(
        id="m4_roberta_focal",
        name="RoBERTa-base + Focal Loss",
        family="transformer",
        huggingface_id="roberta-base",
        use_focal_loss=True,
        description="Best choice for minority classes (sadness, desire).",
        recommended=True,
    ),
    "m5_distilroberta": ModelSpec(
        id="m5_distilroberta",
        name="DistilRoBERTa-base Fine-Tuning",
        family="transformer",
        huggingface_id="distilroberta-base",
        description="40% smaller/faster; good for edge deployment.",
    ),
    "m6_deberta_v3": ModelSpec(
        id="m6_deberta_v3",
        name="DeBERTa-v3-base Fine-Tuning",
        family="transformer",
        huggingface_id="microsoft/deberta-v3-base",
        description="Disentangled attention; often beats RoBERTa on NLU.",
    ),
    "m7_xlm_roberta": ModelSpec(
        id="m7_xlm_roberta",
        name="XLM-RoBERTa-base Fine-Tuning",
        family="transformer",
        huggingface_id="xlm-roberta-base",
        description="Multilingual robustness for mixed-language Reddit text.",
    ),
    "m8_captum_ig": ModelSpec(
        id="m8_captum_ig",
        name="Layer Integrated Gradients (Captum)",
        family="xai",
        description="Token-level attribution; required for production heatmaps.",
        recommended=True,
    ),
}


def get_model(model_id: str) -> ModelSpec:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_id '{model_id}'. Choose from: {list_models()}")
    return MODEL_REGISTRY[model_id]


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def default_transformer_id() -> str:
    return "m4_roberta_focal"


def apply_model_to_config(config: dict, model_id: str) -> dict:
    """Merge model registry settings into training config."""
    spec = get_model(model_id)
    merged = dict(config)
    if spec.huggingface_id:
        merged["model_name"] = spec.huggingface_id
        merged["model_id"] = spec.id
    if spec.family == "transformer":
        merged["use_focal_loss"] = spec.use_focal_loss
    return merged
