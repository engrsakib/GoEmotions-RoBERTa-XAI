# 03 — Kaggle Setup (Full Guide)

Complete step-by-step guide to clone [GoEmotions-RoBERTa-XAI](https://github.com/engrsakib/GoEmotions-RoBERTa-XAI) from the **`main`** branch, attach the GoEmotions dataset, and run the full training pipeline on Kaggle GPU.

---

## Repository

| Item | Value |
|------|-------|
| **GitHub repo** | https://github.com/engrsakib/GoEmotions-RoBERTa-XAI.git |
| **Branch** | `main` |
| **Training code** | `notebooks/scripts/run_pipeline.py` |
| **Kaggle entry** | `notebooks/kaggle/run_training.py` |

---

## Dataset

| Item | Link |
|------|------|
| **Kaggle dataset (use this)** | [shivamb/go-emotions-google-emotions-dataset](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset) |
| **Original source** | [GoEmotions — Google Research](https://github.com/google-research/google-research/tree/master/goemotions) |

The pipeline auto-loads from `/kaggle/input/` when the dataset is attached. **KaggleHub is not used on Kaggle** — attach the dataset as Input or place CSV under `data/raw/goemotions/`.

---

## Step 1 — Create a Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **New Notebook**
3. Configure settings (right sidebar):

| Setting | Value |
|---------|-------|
| **Accelerator** | GPU T4 x2 |
| **Internet** | **ON** (required for `git clone` + HuggingFace) |
| **Persistence** | ON (recommended) |

---

## Step 2 — Add the GoEmotions Dataset

1. In the notebook, click **Add Input** (or **+ Add data**)
2. Search: `go-emotions-google-emotions-dataset`
3. Select **[GoEmotions Google Emotions Dataset](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset)** by shivamb
4. Click **Add**

After adding, data appears at one of:

```
/kaggle/input/notebooks/shivamb/list-of-emotions/   ← notebook input (common)
/kaggle/input/go-emotions-google-emotions-dataset/  ← direct dataset
```

The loader checks **`/kaggle/input/notebooks/shivamb/list-of-emotions`** first.

---

## Step 3 — Clone the Repository (main branch)

Paste this in the **first code cell** and run it:

```python
# Clone main branch from GitHub
!git clone --branch main --depth 1 https://github.com/engrsakib/GoEmotions-RoBERTa-XAI.git /kaggle/working/repo

# Move into notebooks directory
import os
os.chdir("/kaggle/working/repo/notebooks")
print("Working directory:", os.getcwd())
!ls -la
```

Verify you see: `scripts/`, `src/`, `config/`, `kaggle/`, `requirements-train.txt`.

---

## Step 4 — Install Dependencies

Run in a **second code cell**:

```python
!pip install -q -r requirements-train.txt
!python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected output:

```
CUDA: True
GPU: Tesla T4
```

---

## Step 5 — Run the Full Training Pipeline

Run in a **third code cell** (this executes all 7 stages: data → EDA → baselines → train → evaluate → XAI → export):

```python
!python kaggle/run_training.py
```

Or run the main script directly with options:

```python
# Full pipeline — default model M4 (RoBERTa + Focal Loss) + deploy copy
!python scripts/run_pipeline.py --deploy --model-id m4_roberta_focal
```

### What each stage does

| Stage | Output |
|-------|--------|
| 1. Data engineering | `artifacts/processed/train.csv`, `validation.csv`, `test.csv` |
| 2. EDA | `artifacts/figures/class_distribution.png` |
| 3. Baselines | M1 LogReg + M2 SVM macro-F1 scores |
| 4. Train | RoBERTa fine-tuning (4 epochs, ~1–2 h on T4) |
| 5. Evaluate | Test macro-F1, classification report |
| 6. XAI | Captum token heatmaps (5 samples) |
| 7. Export | `artifacts/exports/saved_emotion_model/` |

---

## Step 6 — Run Stages Individually (Optional)

If you prefer to run one stage at a time:

```python
import os
os.chdir("/kaggle/working/repo/notebooks")

# Stage 1 only — data engineering
!python scripts/run_pipeline.py --skip-bootstrap --stage data

# Stage 2 — EDA (requires Stage 1)
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --stage eda

# Stage 3 — baselines only
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --stage baselines

# Stage 4+5 — train + evaluate (requires Stage 1)
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --stage train --model-id m4_roberta_focal

# Stage 6 — XAI validation
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --stage xai

# Stage 7 — export + deploy
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --stage export --deploy
```

---

## Step 7 — Train a Different Model (Optional)

Eight models are registered in `src/training/model_registry.py`:

```python
# DeBERTa-v3 (higher accuracy)
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --model-id m6_deberta_v3 --deploy

# DistilRoBERTa (faster, smaller)
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --model-id m5_distilroberta --deploy

# Standard RoBERTa (no focal loss)
!python scripts/run_pipeline.py --skip-bootstrap --skip-data --model-id m3_roberta_base --deploy
```

See [02-eight-models.md](02-eight-models.md) for all eight models.

---

## Step 8 — Download Trained Weights

After training completes, verify export files:

```python
!ls -la /kaggle/working/repo/notebooks/artifacts/exports/saved_emotion_model/
```

Required files:

- `config.json`
- `model.safetensors` (or `pytorch_model.bin`)
- `tokenizer.json`, `vocab.json`, `merges.txt`
- `label_map.json`

### Download from Kaggle

1. Open **Output** tab in the notebook
2. Download `repo/notebooks/artifacts/exports/saved_emotion_model/`

### Copy to production (local machine)

```bash
# After downloading the export folder locally
cp -r saved_emotion_model/* packages/model/saved_emotion_model/
```

Or publish as a Kaggle Dataset for reuse in future notebooks.

---

## One-Cell Quick Start (Copy-Paste All)

Paste this single cell to clone, install, and train everything:

```python
import os, subprocess, sys

REPO = "https://github.com/engrsakib/GoEmotions-RoBERTa-XAI.git"
REPO_DIR = "/kaggle/working/repo"

# 1. Clone main branch
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--branch", "main", "--depth", "1", REPO, REPO_DIR], check=True)

# 2. Install deps
os.chdir(f"{REPO_DIR}/notebooks")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-train.txt"], check=True)

# 3. Verify GPU
import torch
print(f"CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 4. Run full pipeline (data → train → XAI → export)
subprocess.run([sys.executable, "kaggle/run_training.py"], check=True)

print("\nDone! Weights at: notebooks/artifacts/exports/saved_emotion_model/")
```

---

## Working Directories on Kaggle

| Path | Purpose |
|------|---------|
| `/kaggle/working/repo/` | Cloned GitHub repo (main branch) |
| `/kaggle/working/repo/notebooks/` | Training code root |
| `/kaggle/input/notebooks/shivamb/list-of-emotions/` | GoEmotions CSV (notebook input) |
| `/kaggle/input/go-emotions-google-emotions-dataset/` | GoEmotions CSV (direct dataset) |
| `/kaggle/working/repo/notebooks/artifacts/processed/` | Processed train/val/test CSVs |
| `/kaggle/working/repo/notebooks/artifacts/checkpoints/` | HuggingFace epoch checkpoints |
| `/kaggle/working/repo/notebooks/artifacts/exports/` | Final model for download |

---

## Session Tips

| Issue | Fix |
|-------|-----|
| Session disconnected | Checkpoints saved each epoch — re-run with `--skip-data --stage train` |
| Out of memory (OOM) | Edit `config/train_config.yaml`: `batch_size: 8`, `gradient_accumulation_steps: 2` |
| Slow on CPU | Ensure GPU T4 x2 is selected in notebook settings |
| `git clone` fails | Turn **Internet ON** in notebook settings |
| Dataset not found | Add [GoEmotions dataset](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset) as Input |

---

## Success Criteria

Training succeeded when:

- [ ] Test macro-F1 beats baseline (~0.50)
- [ ] Per-class F1 for sadness & desire > 0.35
- [ ] `artifacts/exports/saved_emotion_model/config.json` exists
- [ ] Captum XAI runs without token/heatmap length errors

See [04-export-checklist.md](04-export-checklist.md) for the full production checklist.

---

## References

- Repository: https://github.com/engrsakib/GoEmotions-RoBERTa-XAI
- Kaggle dataset: https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset
- Methodology: [IEEE_METHODOLOGY.md](IEEE_METHODOLOGY.md)
- Eight models: [02-eight-models.md](02-eight-models.md)
