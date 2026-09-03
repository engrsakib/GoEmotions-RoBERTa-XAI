# GoEmotions Dataset

## Source

| Resource | Link |
|----------|------|
| **Original dataset** | [GoEmotions — Google Research](https://github.com/google-research/google-research/tree/master/goemotions) |
| **Kaggle mirror (recommended)** | [shivamb/go-emotions-google-emotions-dataset](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset) |
| **Training repo** | [engrsakib/GoEmotions-RoBERTa-XAI](https://github.com/engrsakib/GoEmotions-RoBERTa-XAI) |
| **Local path** | `data/raw/goemotions/` (gitignored) |

## Files

| File | Description |
|---|---|
| `*.csv` or `train.tsv` / `dev.tsv` / `test.tsv` | Raw GoEmotions release |

## Columns (merged CSV)

| Column | Type | Notes |
|---|---|---|
| `id` | string | Reddit comment ID |
| `text` | string | Comment text |
| `example_very_unclear` | bool | Drop during preprocessing |
| `admiration` … `neutral` | 0/1 | Multi-label emotion flags (28 emotions) |

## Target Mapping

28 GoEmotions labels are collapsed into 7 production classes. See
[`notebooks/src/data/label_mapping.py`](../notebooks/src/data/label_mapping.py)
and [`packages/model/app/labels.py`](../packages/model/app/labels.py).

## Processed Output

After running the data pipeline, processed splits are written to
`notebooks/artifacts/processed/` (train / validation / test CSVs + stats).
