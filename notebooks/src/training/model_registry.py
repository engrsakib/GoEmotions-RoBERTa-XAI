"""Registry of models/algorithms for GoEmotions classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.training.model_profiles import apply_model_profile

ModelFamily = Literal["baseline", "transformer", "xai"]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    family: ModelFamily
    description: str
    huggingface_id: str | None = None
    use_focal_loss: bool = False
    ignore_mismatched_sizes: bool = False
    recommended: bool = False


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
        description="Single-label with focal loss for minority classes.",
        recommended=True,
    ),
    "m5_distilroberta": ModelSpec(
        id="m5_distilroberta",
        name="DistilRoBERTa-base Fine-Tuning",
        family="transformer",
        huggingface_id="distilroberta-base",
        description="40% smaller/faster; distillation student for deployment.",
    ),
    "m6_deberta_v3": ModelSpec(
        id="m6_deberta_v3",
        name="DeBERTa-v3-base Fine-Tuning",
        family="transformer",
        huggingface_id="microsoft/deberta-v3-base",
        description="IEEE Track A primary; asymmetric loss on 7-macro multi-label.",
        recommended=True,
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
    "m9_twitter_roberta": ModelSpec(
        id="m9_twitter_roberta",
        name="Twitter-RoBERTa-base Fine-Tuning",
        family="transformer",
        huggingface_id="cardiffnlp/twitter-roberta-base",
        description="Social-media pretrained encoder for Reddit/GoEmotions (Paper 3).",
        recommended=True,
    ),
    "m10_deberta_v3_large": ModelSpec(
        id="m10_deberta_v3_large",
        name="DeBERTa-v3-large Fine-Tuning",
        family="transformer",
        huggingface_id="microsoft/deberta-v3-large",
        description="Max-accuracy IEEE Track A; requires 16GB+ GPU.",
    ),
    "m11_twitter_roberta_emotion": ModelSpec(
        id="m11_twitter_roberta_emotion",
        name="Twitter-RoBERTa-base-emotion Fine-Tuning",
        family="transformer",
        huggingface_id="cardiffnlp/twitter-roberta-base-emotion",
        ignore_mismatched_sizes=True,
        description="Emotion-pretrained Twitter encoder; 4-class head replaced with 7.",
    ),
    "m12_bertweet": ModelSpec(
        id="m12_bertweet",
        name="BERTweet-base Fine-Tuning",
        family="transformer",
        huggingface_id="vinai/bertweet-base",
        description="Short social text specialist (Paper 3, 7).",
    ),
    "m13_roberta_large": ModelSpec(
        id="m13_roberta_large",
        name="RoBERTa-large Fine-Tuning",
        family="transformer",
        huggingface_id="roberta-large",
        description="Higher-capacity Track B production encoder (Paper 14).",
    ),
}


def get_model(model_id: str) -> ModelSpec:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_id '{model_id}'. Choose from: {list_models()}")
    return MODEL_REGISTRY[model_id]


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def default_transformer_id() -> str:
    return "m6_deberta_v3"


def apply_model_to_config(config: dict, model_id: str) -> dict:
    """Merge model registry settings and profile overrides into training config."""
    spec = get_model(model_id)
    merged = dict(config)
    if spec.huggingface_id:
        merged["model_name"] = spec.huggingface_id
        merged["model_id"] = spec.id
    if spec.ignore_mismatched_sizes:
        merged["ignore_mismatched_sizes"] = True
    if spec.family == "transformer":
        merged["use_focal_loss"] = spec.use_focal_loss
        if spec.use_focal_loss:
            merged.setdefault("loss_type", "focal")
        else:
            merged.setdefault("loss_type", "weighted_ce")
    merged = apply_model_profile(merged, model_id)
    return merged
