# 03 — Kaggle Setup

## Notebook Settings

| Setting | Value |
|---------|-------|
| **Accelerator** | GPU T4 x2 |
| **Internet** | ON |
| **Persistence** | Save checkpoints each epoch |

## Input Data

Add Kaggle dataset:

```
shivamb/go-emotions-google-emotions-dataset
```

The pipeline auto-detects `/kaggle/input/` paths. If missing, it falls back to
KaggleHub download.

## Bootstrap

See [`scripts/run_pipeline.py`](../scripts/run_pipeline.py) or [`kaggle/run_training.py`](../kaggle/run_training.py).

## Working Directories

| Path | Purpose |
|------|---------|
| `/kaggle/working/repo/notebooks/` | Notebook + src code |
| `/kaggle/working/repo/notebooks/artifacts/processed/` | Processed CSVs |
| `/kaggle/working/repo/notebooks/artifacts/checkpoints/` | HF checkpoints |
| `/kaggle/working/repo/notebooks/artifacts/exports/` | Final model export |

## Output

After training completes, download or publish:

```
notebooks/artifacts/exports/saved_emotion_model/
```

Required files:

- `config.json`
- `model.safetensors` (or `pytorch_model.bin`)
- `tokenizer.json`, `vocab.json`, `merges.txt`
- `label_map.json`

## Session Tips

- Kaggle sessions can disconnect — `save_strategy="epoch"` preserves progress
- Enable FP16 for ~2× speed on T4
- Pin `transformers==4.46.3` (matches production `requirements.txt`)
- If OOM: reduce `batch_size` to 8 and set `gradient_accumulation_steps=2`

## Environment Detection

`src/paths.py` sets paths automatically:

| Environment | `is_kaggle` | Artifacts root |
|-------------|-------------|----------------|
| Kaggle | `True` | `/kaggle/working/.../notebooks/artifacts/` |
| Local | `False` | `<repo>/notebooks/artifacts/` |
