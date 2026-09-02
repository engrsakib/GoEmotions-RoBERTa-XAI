# 02 — Five Deep Learning & Advanced Algorithms

This project uses five algorithms in sequence: one classical baseline, three
advanced training techniques for RoBERTa, and one XAI method aligned with
production.

## Algorithm 1: TF-IDF + Logistic Regression (Baseline)

**Module:** `src/training/baselines.py`

| Parameter | Value |
|-----------|-------|
| Vectorizer | `TfidfVectorizer(max_features=50000, ngram_range=(1,2))` |
| Classifier | `LogisticRegression(class_weight='balanced', max_iter=1000)` |
| Metric | macro-F1 |

**Purpose:** Fast sanity check. If RoBERTa cannot beat this on macro-F1, review
data quality or label mapping before investing GPU time.

---

## Algorithm 2: RoBERTa Fine-Tuning (Primary Deep Model)

**Module:** `src/training/trainer_setup.py`

| Parameter | Value |
|-----------|-------|
| Model | `roberta-base` |
| Head | 7-class sequence classification |
| Max length | 128 |
| Batch size | 16 |
| Epochs | 4 |
| Metric | macro-F1 (not weighted F1) |

**Purpose:** Production-grade transformer encoder fine-tuned on GoEmotions.

---

## Algorithm 3: Focal Loss (Class Imbalance)

**Module:** `src/training/focal_loss.py`

| Parameter | Value |
|-----------|-------|
| Gamma | 2.0 |
| Alpha | inverse class frequency (optional) |

**Purpose:** Down-weight easy majority-class examples (neutral ~42%, anger ~22%)
and focus learning on minority classes (sadness ~4%, desire ~2.5%).

Compare validation macro-F1 against standard cross-entropy; keep the winner.

---

## Algorithm 4: AdamW + Cosine LR Schedule with Warmup

**Module:** `src/training/trainer_setup.py` (via HuggingFace `TrainingArguments`)

| Parameter | Value |
|-----------|-------|
| Optimizer | `adamw_torch` |
| Scheduler | `cosine` |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| FP16 | true (Kaggle T4) |

**Purpose:** Stable fine-tuning without early overshooting on minority classes.
Monitor LR curve in `artifacts/logs/`.

---

## Algorithm 5: Layer Integrated Gradients — Captum (XAI)

**Module:** `src/xai/captum_ig.py`

| Parameter | Value |
|-----------|-------|
| Method | `LayerIntegratedGradients` on embedding layer |
| Steps | 32 |
| Baseline | pad token IDs |

**Purpose:** Token-level attribution heatmaps matching production
`packages/model/app/explainability.py`. Do **not** use `transformers_interpret`.

**Validation:** run on 5–10 test samples; confirm emotion words receive high
attribution scores.

---

## Evaluation Metrics

All models report:

- **macro-F1** (primary selection metric)
- accuracy
- per-class precision / recall / F1
- confusion matrix (RoBERTa test set)

See `src/training/metrics.py`.
