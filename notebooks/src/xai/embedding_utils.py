"""Resolve token embedding layers across HuggingFace encoder architectures."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel

_MODEL_TYPE_TO_BACKBONE = {
    "roberta": "roberta",
    "xlm-roberta": "roberta",
    "deberta": "deberta",
    "deberta-v2": "deberta",
    "bert": "bert",
    "distilbert": "distilbert",
}

_BACKBONE_CANDIDATES = ("deberta", "roberta", "bert", "distilbert")


def get_token_embedding_layer(model: PreTrainedModel) -> torch.nn.Module:
    """Return the token embedding module for Captum LayerIntegratedGradients."""
    model_type = getattr(model.config, "model_type", "") or ""
    attr = _MODEL_TYPE_TO_BACKBONE.get(model_type, model_type.replace("-", "_"))
    backbone = getattr(model, attr, None)

    if backbone is None:
        for candidate in _BACKBONE_CANDIDATES:
            backbone = getattr(model, candidate, None)
            if backbone is not None:
                break

    if backbone is None or not hasattr(backbone, "embeddings"):
        raise AttributeError(
            f"No token embedding layer found for model_type={model_type!r} "
            f"({type(model).__name__})"
        )
    return backbone.embeddings
