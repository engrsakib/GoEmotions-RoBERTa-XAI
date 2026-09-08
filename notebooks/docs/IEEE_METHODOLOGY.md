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
| `none` | No balancing **(IEEE default)** |
| `undersample_neutral` | Random subsample of neutral to ~27% of train |
| `oversample_minority` | Random oversample of below-median classes to median |
| `hybrid` | Undersample neutral + oversample desire/fear |

**Val and test splits are never modified.**

### III-G. Deduplication Policy (`dedup_policy`)

Global pre-split dedup removed ~72% of rows and injected label noise (69% of rows had conflicting labels across duplicate texts). Configurable policies in `src/data/clean.py`:

| Policy | Description |
|--------|-------------|
| `consensus` | Keep duplicate texts only when all copies share the same 7-class label **(default)** |
| `none` | Keep all rows |
| `split_internal` | Dedup within each split after assignment |
| `global_first` | Legacy keep-first (**deprecated**) |

### III-H. Dual-Track Architecture

| Track | Purpose | Labels | Model |
|-------|---------|--------|-------|
| **A — multilabel** | IEEE benchmark (Paper 3, 6) | 7-dim multi-hot macro | DeBERTa-v3 + asymmetric loss |
| **B — singlelabel** | Production API | Clean single-macro rows only | RoBERTa + weighted CE / distillation |

Knowledge distillation (`src/training/distill.py`) transfers Track A teacher logits to Track B student.

**Split modes:** `split_mode: official` (train/dev/test TSV) for papers; `stratified` for application mapping.

**CLI:** `python scripts/run_experiments.py --experiment E2`

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
| Loss (Track A) | Asymmetric clipped loss (Paper 3) |
| Loss (Track B) | Weighted CE or focal (γ=1.0) |
| Train balancing | `none` (default) |
| Dedup policy | `consensus` (default) |
| Track | `singlelabel` (production) or `multilabel` (IEEE) |

When `balance_strategy ≠ none`, class weights are disabled by default to avoid double correction.

**Config file:** `config/train_config.yaml`

### Experiment Matrix (`scripts/run_experiments.py`)

| ID | Track | Model | Loss | Dedup |
|----|-------|-------|------|-------|
| E0 | — | TF-IDF baselines | — | consensus |
| E1 | multilabel | RoBERTa | weighted BCE | consensus |
| E2 | multilabel | DeBERTa-v3 | asymmetric | consensus + official split |
| E3 | singlelabel | RoBERTa focal | weighted CE | consensus |
| E4 | singlelabel | RoBERTa | distillation | consensus |
| E5 | ablation | — | — | none / global_first / hybrid |

---

## VI. Evaluation Protocol

1. Report macro-F1, accuracy, per-class precision/recall/F1  
2. **Argmax baseline** on held-out test set  
3. **Threshold-tuned eval** with optional **minimum precision floor** (e.g. desire ≥ 0.45)  
4. Report **val–test macro-F1 gap** (target < 3%)  
5. **Multi-label metrics** (Track A): macro/micro-F1, Hamming loss, subset accuracy  
6. **XAI faithfulness:** deletion AOPC on N samples (`src/xai/faithfulness.py`)  
7. **External validation:** map SemEval-2018 or curated samples through schema v2 (planned subset in `dataset/external/`)

Thresholds + `uncertain_threshold` exported to `artifacts/exports/saved_emotion_model/thresholds.json`. Production returns `uncertain` when max confidence < τ.

**Implementation:** `src/training/metrics.py`, `src/training/thresholds.py`, `src/training/trainer_setup.py`

---

## VII. Explainability (XAI)

Layer Integrated Gradients (`n_steps=32`) plus **deletion AOPC** faithfulness metric (Paper 13). Implementation: `src/xai/captum_ig.py`, `src/xai/faithfulness.py`.

---

## VIII. Deployment

Export checkpoint to `artifacts/exports/saved_emotion_model/`. Inference loads `thresholds.json` and supports uncertain rejection (`packages/model/app/inference.py`).

---

## IX. Reproducibility

```bash
cd notebooks
pip install -r requirements-train.txt
python scripts/audit_label_mapping.py
python scripts/run_experiments.py --experiment E3
```

Kaggle:

```bash
python kaggle/run_training.py --force-data
```

See also [papers/README.md](../papers/README.md) for the 16-paper bibliography.

---

## References

[1] D. Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions," ACL, 2020.  
[2] Y. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," 2019.  
[3] T.-Y. Lin et al., "Focal Loss for Dense Object Detection," ICCV, 2017.  
[4] M. Sundararajan et al., "Axiomatic Attribution for Deep Networks," ICML, 2017.  
[5] P. He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention," ICLR, 2021.
