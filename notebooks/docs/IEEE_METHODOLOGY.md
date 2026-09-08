# IEEE-Style Methodology Document

**Title:** GoEmotions-RoBERTa-XAI: A Reproducible Pipeline for Seven-Class Emotion Classification with Token-Level Explainability

**Author:** Md. Nazmus Sakib  
**Repository:** [GoEmotions-RoBERTa-XAI](https://github.com/engrsakib/GoEmotions-RoBERTa-XAI)

---

## Abstract

This document describes the methodology for fine-tuning transformer encoders on the GoEmotions dataset mapped to seven production emotion categories, with classical baselines, imbalance-aware loss functions, train-only class balancing, validation-set threshold tuning, and Captum Integrated Gradients for token-level attribution heatmaps deployed via FastAPI.

---

## I. Introduction

Reddit comment emotion detection requires handling multi-label annotations, severe class imbalance, and explainable predictions for user-facing heatmaps. This pipeline separates **training** (`notebooks/`) from **inference** (`packages/model/`).

---

## II. Dataset

| Property | Value |
|----------|-------|
| Source | GoEmotions (Google Research) |
| Kaggle mirror | `shivamb/go-emotions-google-emotions-dataset` |
| Raw samples | 211,225 |
| Features | `id`, `text`, 28 emotion flags, `example_very_unclear` |
| Target classes | 7 (aligned with `packages/model/app/labels.py`) |
| Label schema | **v2.0** (full 28→7 coverage) |

---

## III. Data Engineering Methodology

### III-A. Label Mapping (Schema v2.0)

All twenty-eight GoEmotions emotion columns map to exactly one of seven production IDs. Multi-label rows resolve by priority list `[0,1,2,3,4,5,6]` (last active group wins).

| Macro ID | Production label | Source emotions | Rationale |
|----------|------------------|-----------------|-----------|
| 0 | neutral | neutral, confusion, curiosity, realization, surprise | Cognitive / low-arousal / ambiguous |
| 1 | sadness_grief | sadness, grief, disappointment, remorse | Negative self/other evaluation |
| 2 | joy_amusement_excitement_optimism | joy, amusement, excitement, optimism, admiration, approval, gratitude, pride, relief | Positive affect |
| 3 | anger_annoyance_disapproval_disgust | anger, annoyance, disapproval, disgust | Hostile / negative social |
| 4 | desire | desire | Romantic / wanting |
| 5 | fear_nervousness | fear, nervousness, embarrassment | Anxiety / threat / social discomfort |
| 6 | love | love, caring | Affiliative warmth |

`validate_mapping()` asserts no unmapped columns and no overlapping assignments. Audit script: `scripts/audit_label_mapping.py`.

### III-B. Filtering

Rows removed when: `example_very_unclear=True`, no mappable label, empty text, `char_length < 3`, or `token_length_approx > 128`.

### III-C. Deduplication

Normalized lowercase text deduplication occurs **before** splitting to prevent leakage.

### III-D. Splitting

Stratified 80/10/10 train/validation/test split (`random_state=42`).

### III-E. Validation Gates

- Zero cross-split text overlap  
- Class proportion spread ≤ 0.5% across splits  
- Minimum 100 training samples per class  

### III-F. Train-Only Class Balancing

To reduce majority-class gradient dominance without distorting evaluation:

| Strategy | Description |
|----------|-------------|
| `none` | No balancing (ablation) |
| `undersample_neutral` | Random subsample of neutral to ~27% of train |
| `oversample_minority` | Random oversample of below-median classes to median |
| `hybrid` | Undersample neutral + oversample desire/fear (default) |

**Val and test splits are never modified.** Config: `balance_strategy`, `target_neutral_pct`, `balance_random_seed`.

**Implementation:** `src/data/balance.py`, wired in `src/data/pipeline.py`  
**CLI:** `python scripts/run_pipeline.py --force-data --stage data`

---

## IV. Model Suite (Eight Algorithms)

| ID | Algorithm | Role |
|----|-----------|------|
| M1 | TF-IDF + Logistic Regression | Classical baseline |
| M2 | TF-IDF + Linear SVM | Linear baseline |
| M3 | RoBERTa-base + Weighted CE | Production encoder (balanced loss) |
| M4 | RoBERTa-base + Focal Loss | **Recommended** (imbalance) |
| M5 | DistilRoBERTa-base | Efficient deployment |
| M6 | DeBERTa-v3-base | Accuracy alternative |
| M7 | XLM-RoBERTa-base | Multilingual robustness |
| M8 | Captum Integrated Gradients | XAI heatmaps |

See [02-eight-models.md](02-eight-models.md) for selection guidance.

---

## V. Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Max sequence length | 128 |
| Batch size | 16 |
| Epochs | 5 (upper bound) |
| Early stopping | patience=2 on `eval_macro_f1` |
| Learning rate | 2×10⁻⁵ |
| Optimizer | AdamW |
| LR schedule | Cosine + 10% warmup |
| Primary metric | Macro-F1 |
| Loss (M4) | Focal (γ=1.5), optional class weights |
| Loss (M3) | Weighted cross-entropy (`sqrt_inverse` weights) |
| Train balancing | `hybrid` (default) |

When `balance_strategy ≠ none`, class weights are disabled by default to avoid double correction.

**Config file:** `config/train_config.yaml`

### Recommended Experiment Matrix

| Run | Model | Loss | γ | Balance |
|-----|-------|------|---|---------|
| A | M3 | weighted_ce | — | hybrid |
| B | M4 | focal | 1.5 | hybrid |
| C | M4 | focal | 1.0 | hybrid |

Select the configuration with best **validation** macro-F1 before final test evaluation.

---

## VI. Evaluation Protocol

1. Report macro-F1, accuracy, per-class precision/recall/F1  
2. **Argmax baseline** on held-out test set  
3. **Threshold-tuned eval:** per-class probability thresholds optimized on validation (coordinate ascent, step=0.05), applied once to test  
4. Report **val–test macro-F1 gap** as overfitting check (target < 3%)  
5. Compare transformer against M1/M2 baselines  
6. Success criteria: macro-F1 > baseline; sadness/desire F1 > 0.35  

Thresholds exported to `artifacts/exports/saved_emotion_model/thresholds.json`.

**Implementation:** `src/training/metrics.py`, `src/training/thresholds.py`, `src/training/trainer_setup.py`

---

## VII. Explainability (XAI)

Layer Integrated Gradients applied to RoBERTa embedding layer (`n_steps=32`), matching production `packages/model/app/explainability.py`.

---

## VIII. Deployment

Export checkpoint to `artifacts/exports/saved_emotion_model/`, copy to `packages/model/saved_emotion_model/`, serve via Docker Compose (ports 3000/4000/8000).

---

## IX. Reproducibility

```bash
cd notebooks
pip install -r requirements-train.txt
python scripts/audit_label_mapping.py
python scripts/run_pipeline.py --force-data --deploy
```

Kaggle:

```bash
python kaggle/run_training.py --force-data
```

---

## References

[1] D. Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions," ACL, 2020.  
[2] Y. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," 2019.  
[3] T.-Y. Lin et al., "Focal Loss for Dense Object Detection," ICCV, 2017.  
[4] M. Sundararajan et al., "Axiomatic Attribution for Deep Networks," ICML, 2017.  
[5] P. He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention," ICLR, 2021.
