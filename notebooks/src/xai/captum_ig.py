"""Captum Integrated Gradients — mirrors production explainability.py."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    max_abs = float(np.max(np.abs(values)))
    if max_abs == 0:
        return values
    return values / max_abs


def _format_token(token: str) -> str:
    if token.startswith("Ġ"):
        return token[1:]
    return token


def compute_integrated_gradients(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    target_class: int,
    device: torch.device,
    max_length: int = 128,
    n_steps: int = 32,
) -> Tuple[List[str], List[float]]:
    model.eval()
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    def forward_func(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs = model(input_ids=ids, attention_mask=mask)
        return outputs.logits[:, target_class]

    embedding_layer = model.roberta.embeddings
    lig = LayerIntegratedGradients(forward_func, embedding_layer)

    pad_id = tokenizer.pad_token_id or 1
    baselines = torch.full_like(input_ids, pad_id)

    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baselines,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        internal_batch_size=1,
    )

    token_attr = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    ids = input_ids.squeeze(0).tolist()
    mask = attention_mask.squeeze(0).tolist()

    tokens: List[str] = []
    scores: List[float] = []
    special = {tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token}

    for idx, token_id in enumerate(ids):
        if idx >= len(token_attr) or mask[idx] == 0:
            continue
        raw_token = tokenizer.convert_ids_to_tokens(token_id)
        if raw_token in special:
            continue
        tokens.append(_format_token(raw_token))
        scores.append(float(token_attr[idx]))

    normalized = _normalize_scores(np.array(scores))
    return tokens, normalized.tolist()


def explain_samples(
    model,
    tokenizer,
    samples: list[str],
    device: torch.device,
    max_length: int = 128,
    n_steps: int = 32,
) -> list[dict]:
    results = []
    model.to(device)
    model.eval()

    for text in samples:
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length, padding=True
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            target_class = int(torch.argmax(logits, dim=-1).item())

        tokens, heatmap = compute_integrated_gradients(
            model, tokenizer, text, target_class, device, max_length, n_steps
        )
        results.append(
            {
                "text": text,
                "target_class": target_class,
                "tokens": tokens,
                "heatmap": heatmap,
            }
        )
    return results
