# 01 — Data Engineering

## Overview

This pipeline transforms raw GoEmotions multi-label data into stratified
train / validation / test CSVs with 7 single-label classes aligned to
production inference.

## Pipeline Steps

### 1. Load (`src/data/load.py`)

- **Kaggle input:** scan `/kaggle/input/` for GoEmotions CSV/TSV
- **KaggleHub fallback:** `kagglehub.dataset_download("shivamb/go-emotions-google-emotions-dataset")`
- **Local:** `data/raw/goemotions/*.csv`
- **Official splits:** if `train.tsv`, `dev.tsv`, `test.tsv` exist, load separately

Audit outputs: row count, null counts, `example_very_unclear` rate, text length stats.

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
| `label_map.json` | id2label / label2id (matches production) |
| `data_stats.json` | Counts, percentages, cleaning log |

## Validation Gates

Before training, verify:

- [ ] No duplicate normalized texts across train / val / test
- [ ] Class proportions within ±0.5% across splits
- [ ] Minimum 100 training samples per class (watch class 4 desire)
- [ ] `label_map.json` matches `packages/model/saved_emotion_model/label_map.json`

## Run from Notebook

```python
from src.data.pipeline import run_data_pipeline

result = run_data_pipeline()
print(result["stats"])
```

## Expected Volumes

| Stage | Approx rows |
|-------|-------------|
| Raw CSV | ~211,000 |
| After label filter | ~138,000 |
| After dedup | ~51,000 |
| Train (80%) | ~41,000 |
