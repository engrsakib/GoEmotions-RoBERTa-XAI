"""XAI faithfulness metrics (insertion/deletion AOPC) — Paper 13."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def predict_proba_fn(model, tokenizer, text: str, target_class: int, device, max_length: int = 128) -> float:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]
    return float(probs[target_class].item())


def deletion_aopc(
    model,
    tokenizer,
    text: str,
    tokens: list[str],
    attributions: list[float],
    target_class: int,
    device,
    max_length: int = 128,
    steps: int = 10,
) -> float:
    """
    Area Over the Perturbation Curve — delete top-attributed tokens progressively.
    Higher score = attributions align with model behavior.
    """
    if not tokens or not attributions:
        return 0.0

    baseline = predict_proba_fn(model, tokenizer, text, target_class, device, max_length)
    ranked_indices = np.argsort(attributions)[::-1]
    step_size = max(1, len(ranked_indices) // steps)

    scores = [baseline]
    words = text.split()
    for i in range(1, steps + 1):
        remove_count = min(i * step_size, len(ranked_indices))
        remove_set = set(ranked_indices[:remove_count])
        perturbed = " ".join(
            tok for idx, tok in enumerate(words) if idx not in remove_set
        ) or "[EMPTY]"
        scores.append(predict_proba_fn(model, tokenizer, perturbed, target_class, device, max_length))

    scores = np.array(scores)
    return float(np.trapz(baseline - scores, dx=1.0 / steps))


def evaluate_faithfulness_batch(
    model,
    tokenizer,
    samples: list[str],
    target_classes: list[int],
    explain_fn: Callable,
    device,
    max_length: int = 128,
) -> dict:
    """
    Run IG (or other) explain_fn and compute mean deletion AOPC.
    explain_fn(model, tokenizer, text, target_class, device) -> (tokens, heatmap)
    """
    aopcs = []
    for text, target_class in zip(samples, target_classes):
        tokens, heatmap = explain_fn(model, tokenizer, text, target_class, device)
        score = deletion_aopc(
            model, tokenizer, text, tokens, heatmap, target_class, device, max_length
        )
        aopcs.append(score)

    return {
        "mean_deletion_aopc": float(np.mean(aopcs)) if aopcs else 0.0,
        "std_deletion_aopc": float(np.std(aopcs)) if aopcs else 0.0,
        "n_samples": len(aopcs),
        "per_sample_aopc": aopcs,
    }
