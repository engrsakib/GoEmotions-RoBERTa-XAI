# 01 — Data Engineering

## Overview

This pipeline transforms raw GoEmotions multi-label data into stratified
train / validation / test CSVs with 7 single-label classes aligned to
production inference.

## Pipeline Steps

### 1. Load (`src/data/load.py`)

- **Kaggle input:** checks known mount paths first, then walks `/kaggle/input/` for GoEmotions CSV/TSV
- **Local:** `data/raw/goemotions/*.csv`
- **KaggleHub:** local development only — **not used on Kaggle** (avoids BackendError)

**Split strategy:** the default pipeline uses a **custom stratified 80/10/10 split** on the merged CSV. The helper `load_official_splits()` exists for official `train.tsv` / `dev.tsv` / `test.tsv` files but is **not wired into `run_data_pipeline()`** today.

Audit outputs (logged via `src/data/reporting.py`): row count, null counts, `example_very_unclear` rate, text length stats.

### 2. Label Mapping (`src/data/label_mapping.py`)

28 GoEmotions columns → 7 target IDs:

| ID | Production label | Source emotions |
|----|------------------|-----------------|
| 0 | neutral | neutral |
| 1 | sadness_grief | sadness, grief |
| 2 | joy_amusement_excitement_optimism | joy, amusement, excitement, optimism |
| 3 | anger_annoyance_disapproval_disgust | anger, annoyance, disapproval, disgust |
| 4 | desire | desire |
| 5 | fear_nervousness | fear, nervousness |
| 6 | love | love |

**Multi-label resolution:** when multiple target groups are active, the highest
priority ID wins (order: 0→6, last wins).

**Rows dropped:**
- `example_very_unclear == True`
- No mappable emotion
- Empty / whitespace-only text

### 3. Text Cleaning (`src/data/clean.py`)

Applied **before splitting** to prevent leakage:

1. Strip whitespace, coerce to string
2. UTF-8 normalization (drop invalid bytes)
3. Collapse repeated whitespace
4. Drop `char_length < 3`
5. Drop `token_length_approx > 128`
6. Deduplicate on normalized lowercase text (keep first)

### 4. Stratified Split (`src/data/split.py`)

- 80% train / 10% val / 10% test (seed=42)
- Stratify on `encoded_label`
- Reset index on all splits

### 5. Export

Written to `notebooks/artifacts/processed/`:

| File | Contents |
|------|----------|
| `train.csv` | Training split |
| `validation.csv` | Validation split |
| `test.csv` | Test split |
| `label_map.json` | id2label, label2id, target_id_to_emotions, schema_version |
| `data_stats.json` | Counts, percentages, cleaning log, audit metadata |

## Dataset Metadata Logging

Human-readable logs are emitted by [`src/data/reporting.py`](../src/data/reporting.py):

- **Label schema** — 7 production classes and source GoEmotions columns
- **Raw audit** — row count, column count, unclear rate, avg text length
- **Mapping / cleaning / split** — row deltas and per-class counts with label names
- **Validation** — leakage check and class balance tolerance

On CLI runs, metadata prints during Stage 1. With `--skip-data`, stats are **replayed from `data_stats.json`**.

```bash
python scripts/run_pipeline.py --skip-bootstrap --stage data
```

Or programmatically:

```python
from src.data.pipeline import run_data_pipeline
from src.data.reporting import log_pipeline_stats

result = run_data_pipeline()
log_pipeline_stats(result["stats"])
```

## Validation Gates

Before training, verify:

- [ ] No duplicate normalized texts across train / val / test
- [ ] Class proportions within ±0.5% across splits
- [ ] Minimum 100 training samples per class (watch class 4 desire)
- [ ] `label_map.json` matches `packages/model/saved_emotion_model/label_map.json`

## Run from CLI

Primary entry point (replaces `lab_final.ipynb`):

```bash
python scripts/run_pipeline.py --skip-bootstrap --stage data
```

Data-only script:

```bash
python scripts/01_data_engineering.py
```

## Expected Volumes

| Stage | Approx rows |
|-------|-------------|
| Raw CSV | ~211,000 |
| After label filter | ~138,000 |
| After dedup | ~51,000 |
| Train (80%) | ~41,000 |
