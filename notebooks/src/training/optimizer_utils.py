"""AdamW parameter groups with Bias/LayerNorm weight-decay exclusion."""

from __future__ import annotations

NO_DECAY_SUFFIXES = ("bias", "LayerNorm.weight", "layer_norm.weight")


def get_no_decay_parameter_names(model) -> set[str]:
    """Return parameter names excluded from weight decay (DeBERTa/RoBERTa standard)."""
    no_decay: set[str] = set()
    for name, _param in model.named_parameters():
        if any(name.endswith(suffix) for suffix in NO_DECAY_SUFFIXES):
            no_decay.add(name)
    return no_decay


def build_adamw_param_groups(model, weight_decay: float) -> list[dict]:
    """
    Build two AdamW groups: decay for weights, zero decay for bias + LayerNorm.
    """
    no_decay = get_no_decay_parameter_names(model)
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
