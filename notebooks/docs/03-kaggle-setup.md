# 03 — Kaggle Setup Guide

End-to-end instructions to train **GoEmotions-RoBERTa-XAI** on Kaggle GPU.  
Clone the latest `main` branch, attach the GoEmotions dataset, install dependencies, and run the full pipeline.

**Repository:** [github.com/engrsakib/GoEmotions-RoBERTa-XAI](https://github.com/engrsakib/GoEmotions-RoBERTa-XAI) · **Branch:** `main`

---

## Before You Start (Checklist)

Complete these steps **before** running any training code:

| # | Task | Required |
|---|------|----------|
| 1 | Create a Kaggle notebook with **GPU T4 x2** | Yes |
| 2 | Turn **Internet ON** (Settings → Internet) | Yes |
| 3 | Add GoEmotions dataset as **Input** (Step 2 below) | Yes |
| 4 | Run **Cell 1** — clean clone + verify layout | Yes |
| 5 | Run **Cell 2** — install dependencies + check CUDA | Yes |
| 6 | Run **Cell 3** — full training pipeline | Yes |

> **Re-runs:** Always use the clean-clone cell (Cell 1) when restarting a session. It removes a stale `/kaggle/working/repo` so you pull the latest code from GitHub.

---

## Repository Reference

| Item | Value |
|------|-------|
| **GitHub URL** | `https://github.com/engrsakib/GoEmotions-RoBERTa-XAI.git` |
| **Branch** | `main` |
| **Clone target** | `/kaggle/working/repo` |
| **Working directory** | `/kaggle/working/repo/notebooks` |
| **Main orchestrator** | `scripts/run_pipeline.py` |
| **Kaggle entry point** | `kaggle/run_training.py` |

---

## Step 1 — Create a Kaggle Notebook

1. Open [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Configure the sidebar:

| Setting | Value |
|---------|-------|
| **Accelerator** | GPU T4 x2 |
| **Internet** | **ON** (required for `git clone` and HuggingFace) |
| **Persistence** | ON (recommended) |

---

## Step 2 — Add the GoEmotions Dataset

1. Click **Add Input** (or **+ Add data**)
2. Search: `go-emotions-google-emotions-dataset`
3. Select **[GoEmotions Google Emotions Dataset](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset)** (author: shivamb)
4. Click **Add** → **Save Version** (required for Input to mount)

The pipeline auto-discovers CSV/TSV files under `/kaggle/input/`. Common mount paths:

```
/kaggle/input/list-of-emotions/
/kaggle/input/notebooks/shivamb/list-of-emotions/
/kaggle/input/go-emotions-google-emotions-dataset/
```

**KaggleHub is not used on Kaggle** — attach the dataset as Input or place a CSV under `data/raw/goemotions/`.

### Verify dataset (optional)

Run before training to confirm the Input is mounted:

```python
from pathlib import Path

for path in Path("/kaggle/input").rglob("*.csv"):
    print(path)
```

You should see at least one GoEmotions CSV path.

---

## Step 3 — Cell 1: Clean Clone and Verify Layout

Paste into the **first code cell**. This removes any previous clone, fetches fresh `main`, and confirms the project layout.

```python
import os
import shutil

REPO_URL = "https://github.com/engrsakib/GoEmotions-RoBERTa-XAI.git"
REPO_DIR = "/kaggle/working/repo"
NOTEBOOKS_DIR = f"{REPO_DIR}/notebooks"

# Remove stale clone so every run uses the latest main
if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
    print(f"Cleaned previous directory: {REPO_DIR}")

# Shallow clone — main branch only
!git clone --branch main --depth 1 {REPO_URL} {REPO_DIR}

os.chdir(NOTEBOOKS_DIR)
print("Updated Working Directory:", os.getcwd())

!ls -la
```

**Expected output** — you should see these directories/files:

| Path | Purpose |
|------|---------|
| `scripts/` | Pipeline CLI (`run_pipeline.py`) |
| `src/` | Data, training, XAI modules |
| `config/` | `train_config.yaml` |
| `kaggle/` | Kaggle entry script |
| `requirements-train.txt` | Python dependencies |

If any of the above is missing, check Internet is ON and re-run Cell 1.

---

## Step 4 — Cell 2: Install Dependencies and Verify GPU

```python
!pip install -q -r requirements-train.txt

import torch

cuda_ok = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "none"
print(f"CUDA available: {cuda_ok}")
print(f"GPU: {gpu_name}")
```

**Expected output:**

```
CUDA available: True
GPU: Tesla T4
```

If `CUDA available: False`, open notebook Settings and select **GPU T4 x2**, then restart the session.

---

## Step 5 — Cell 3: Run the Full Training Pipeline

Runs all stages: data → EDA → baselines → train → evaluate → XAI → export.

```python
import os

os.chdir("/kaggle/working/repo/notebooks")
!python kaggle/run_training.py
```

**Alternative** — explicit model and deploy flag:

```python
!python scripts/run_pipeline.py --skip-bootstrap --deploy --model-id m4_roberta_focal
```

### Pipeline stages

| Stage | Output |
|-------|--------|
| 1. Data engineering | `artifacts/processed/train.csv`, `validation.csv`, `test.csv` |
| 2. EDA | `artifacts/figures/class_distribution.png` |
| 3. Baselines | M1 LogReg + M2 SVM macro-F1 |
| 4. Train | RoBERTa fine-tuning (~1–2 h on T4, 4 epochs) |
| 5. Evaluate | Test macro-F1, classification report |
| 6. XAI | Captum token heatmaps (5 samples) |
| 7. Export | `artifacts/exports/saved_emotion_model/` |

---

## Step 6 — Run Stages Individually (Optional)

Use after Cell 1–2 when you want fine-grained control:

```python
import os

os.chdir("/kaggle/working/repo/notebooks")

# Stage 1 — data engineering only
!python scripts/run_pipeline.py --skip-bootstrap --stage data

# Stage 2 — EDA (requires processed splits)
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --stage eda

# Stage 3 — baselines
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --stage baselines

# Stage 4+5 — train + evaluate
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --stage train --model-id m4_roberta_focal

# Stage 6 — XAI
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --stage xai

# Stage 7 — export + deploy copy
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --stage export --deploy
```

> If `--skip-data` is set but `train.csv` is missing, the pipeline **automatically re-runs Stage 1** instead of crashing.

---

## Step 7 — Train a Different Model (Optional)

Eight models are registered in `src/training/model_registry.py`. Examples:

```python
# DeBERTa-v3 (higher accuracy)
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --model-id m6_deberta_v3 --deploy

# DistilRoBERTa (faster, smaller)
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --model-id m5_distilroberta --deploy

# Standard RoBERTa (no focal loss)
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --model-id m3_roberta_base --deploy
```

See [02-eight-models.md](02-eight-models.md) for the full model list.

---

## Step 8 — Download Trained Weights

Verify export artifacts:

```python
!ls -la /kaggle/working/repo/notebooks/artifacts/exports/saved_emotion_model/
```

Required files:

- `config.json`
- `model.safetensors` (or `pytorch_model.bin`)
- `tokenizer.json`, `vocab.json`, `merges.txt`
- `label_map.json`

### Download from Kaggle

1. Open the notebook **Output** tab
2. Download `repo/notebooks/artifacts/exports/saved_emotion_model/`

### Copy to production (local machine)

```bash
cp -r saved_emotion_model/* packages/model/saved_emotion_model/
```

Or publish the export folder as a Kaggle Dataset for reuse.

---

## One-Cell Quick Start (Advanced)

Single cell for experienced users — clean clone, install, train, and print export path:

```python
import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/engrsakib/GoEmotions-RoBERTa-XAI.git"
REPO_DIR = "/kaggle/working/repo"
NOTEBOOKS_DIR = f"{REPO_DIR}/notebooks"

if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
    print(f"Cleaned previous directory: {REPO_DIR}")

subprocess.run(
    ["git", "clone", "--branch", "main", "--depth", "1", REPO_URL, REPO_DIR],
    check=True,
)

os.chdir(NOTEBOOKS_DIR)
print("Working directory:", os.getcwd())

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-train.txt"], check=True)

import torch
print(f"CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

subprocess.run([sys.executable, "kaggle/run_training.py"], check=True)

print("\nDone. Weights: /kaggle/working/repo/notebooks/artifacts/exports/saved_emotion_model/")
```

---

## Working Directories on Kaggle

| Path | Purpose |
|------|---------|
| `/kaggle/working/repo/` | Cloned GitHub repo (`main`) |
| `/kaggle/working/repo/notebooks/` | Training code root (run all commands here) |
| `/kaggle/input/list-of-emotions/` | GoEmotions CSV (direct slug mount) |
| `/kaggle/input/notebooks/shivamb/list-of-emotions/` | GoEmotions CSV (notebook input) |
| `/kaggle/input/go-emotions-google-emotions-dataset/` | GoEmotions CSV (direct dataset) |
| `/kaggle/working/repo/notebooks/artifacts/processed/` | Processed train/val/test CSVs |
| `/kaggle/working/repo/notebooks/artifacts/checkpoints/` | HuggingFace epoch checkpoints |
| `/kaggle/working/repo/notebooks/artifacts/exports/` | Final model for download |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Stale code after GitHub update | Re-run **Cell 1** (clean clone removes old `/kaggle/working/repo`) |
| Session disconnected | Checkpoints saved each epoch — re-run Cell 1–2, then `--skip-data --stage train` |
| `train.csv` not found with `--skip-data` | Attach GoEmotions Input; pipeline auto-runs Stage 1 if splits are missing |
| Out of memory (OOM) | Edit `config/train_config.yaml`: `batch_size: 8`, `gradient_accumulation_steps: 2` |
| Slow on CPU | Enable **GPU T4 x2** in notebook settings and restart |
| `git clone` fails | Turn **Internet ON** in notebook settings |
| Dataset not found | Add [GoEmotions dataset](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset) as Input and **Save Version** |

---

## Success Criteria

Training succeeded when:

- [ ] Test macro-F1 beats baseline (~0.50)
- [ ] Per-class F1 for sadness and desire > 0.35
- [ ] `artifacts/exports/saved_emotion_model/config.json` exists
- [ ] Captum XAI runs without token/heatmap length errors

See [04-export-checklist.md](04-export-checklist.md) for the production deployment checklist.

---

## Acknowledgements and References

| Resource | Link |
|----------|------|
| **This repository** | https://github.com/engrsakib/GoEmotions-RoBERTa-XAI |
| **GoEmotions (Kaggle mirror)** | https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset |
| **GoEmotions (original)** | https://github.com/google-research/google-research/tree/master/goemotions |
| **Methodology** | [IEEE_METHODOLOGY.md](IEEE_METHODOLOGY.md) |
| **Model registry** | [02-eight-models.md](02-eight-models.md) |
| **Data pipeline** | [01-data-engineering.md](01-data-engineering.md) |

**Dataset citation:** GoEmotions dataset (Demszky et al., 2020) via [shivamb/go-emotions-google-emotions-dataset](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset) on Kaggle.

**Base models:** HuggingFace transformers (`roberta-base`, etc.) — see `config/train_config.yaml` and [02-eight-models.md](02-eight-models.md).
