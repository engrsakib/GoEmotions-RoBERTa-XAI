# IEEE-Style Methodology Document

**Title:** GoEmotions-RoBERTa-XAI: A Reproducible Pipeline for Seven-Class Emotion Classification with Token-Level Explainability

**Author:** Md. Nazmus Sakib  
**Repository:** [GoEmotions-RoBERTa-XAI](https://github.com/engrsakib/GoEmotions-RoBERTa-XAI)

---

## Abstract

This document describes the methodology for fine-tuning transformer encoders on the GoEmotions dataset mapped to seven production emotion categories, with classical baselines, imbalance-aware loss functions, and Captum Integrated Gradients for token-level attribution heatmaps deployed via FastAPI.

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

---

## III. Data Engineering Methodology

### III-A. Label Mapping

Twenty-eight GoEmotions labels collapse into seven IDs (0–6). Multi-label rows resolve by priority list `[0,1,2,3,4,5,6]` (last active group wins).

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

**Implementation:** `src/data/pipeline.py`  
**CLI:** `python scripts/01_data_engineering.py`

---

## IV. Model Suite (Eight Algorithms)

| ID | Algorithm | Role |
|----|-----------|------|
| M1 | TF-IDF + Logistic Regression | Classical baseline |
| M2 | TF-IDF + Linear SVM | Linear baseline |
| M3 | RoBERTa-base | Production encoder |
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
| Epochs | 4 |
| Learning rate | 2×10⁻⁵ |
| Optimizer | AdamW |
| LR schedule | Cosine + 10% warmup |
| Primary metric | Macro-F1 |
| Loss (M4) | Focal (γ=2.0) + class weights |

**Config file:** `config/train_config.yaml`

---

## VI. Evaluation Protocol

1. Report macro-F1, accuracy, per-class precision/recall/F1  
2. Compare transformer against M1/M2 baselines  
3. Success criteria: macro-F1 > baseline; sadness/desire F1 > 0.35  
4. Confusion matrix on held-out test set (5,151 samples)

**Implementation:** `src/training/metrics.py`

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
python scripts/run_pipeline.py --deploy
```

Kaggle:

```bash
python kaggle/run_training.py
```

---

## References

[1] D. Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions," ACL, 2020.  
[2] Y. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," 2019.  
[3] T.-Y. Lin et al., "Focal Loss for Dense Object Detection," ICCV, 2017.  
[4] M. Sundararajan et al., "Axiomatic Attribution for Deep Networks," ICML, 2017.  
[5] P. He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention," ICLR, 2021.
