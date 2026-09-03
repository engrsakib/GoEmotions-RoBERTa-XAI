# GoEmotions-RoBERTa Training Pipeline

Professional Python pipeline for training on **Kaggle GPU** or locally.  
Production inference lives in [`packages/model/`](../packages/model/).

**Repository:** https://github.com/engrsakib/GoEmotions-RoBERTa-XAI.git (branch: `main`)

---

## Quick Start (Kaggle GPU) — Recommended

Full step-by-step guide: **[docs/03-kaggle-setup.md](docs/03-kaggle-setup.md)**

### 1. Create Kaggle notebook

- Accelerator: **GPU T4 x2**
- Internet: **ON**
- Add dataset: [GoEmotions Google Emotions Dataset](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset)

### 2. Clone main branch and train (one cell)

```python
import os, subprocess, sys

REPO = "https://github.com/engrsakib/GoEmotions-RoBERTa-XAI.git"
REPO_DIR = "/kaggle/working/repo"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--branch", "main", "--depth", "1", REPO, REPO_DIR], check=True)

os.chdir(f"{REPO_DIR}/notebooks")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-train.txt"], check=True)
subprocess.run([sys.executable, "kaggle/run_training.py"], check=True)
```

### 3. Download weights

After training: `artifacts/exports/saved_emotion_model/` → copy to `packages/model/saved_emotion_model/`

---

## Quick Start (Local)

```bash
git clone --branch main https://github.com/engrsakib/GoEmotions-RoBERTa-XAI.git
cd GoEmotions-RoBERTa-XAI/notebooks
pip install -r requirements-train.txt
python scripts/run_pipeline.py --deploy
```

---

## Dataset

| Source | Link |
|--------|------|
| **Kaggle** | [shivamb/go-emotions-google-emotions-dataset](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset) |
| **Original** | [GoEmotions — Google Research](https://github.com/google-research/google-research/tree/master/goemotions) |
| **Local path** | `data/raw/goemotions/` (gitignored) |

---

## Directory Layout

```
notebooks/
├── scripts/
│   ├── run_pipeline.py           # Main orchestrator (Stages 0–7)
│   └── 01_data_engineering.py    # Data-only stage
├── src/
│   ├── bootstrap/                # Kaggle/local environment
│   ├── data/                     # Load, map, clean, split
│   ├── training/                 # Baselines, transformers, registry
│   ├── visualization/            # EDA figures
│   └── xai/                      # Captum Integrated Gradients
├── config/train_config.yaml
├── docs/
│   ├── 03-kaggle-setup.md        # Full Kaggle guide
│   ├── IEEE_METHODOLOGY.md
│   └── 02-eight-models.md
├── kaggle/
│   ├── run_training.py           # Kaggle entry point
│   └── kernel-metadata.json
└── artifacts/                    # gitignored outputs
```

---

## Pipeline Stages

| Stage | Command | Description |
|-------|---------|-------------|
| 0 | (auto) | Bootstrap environment |
| 1 | `--stage data` | Data engineering |
| 2 | `--stage eda` | Class distribution figure |
| 3 | `--stage baselines` | M1 LogReg + M2 SVM |
| 4–5 | `--stage train` | Transformer fine-tune + evaluate |
| 6 | `--stage xai` | Captum heatmaps (M8) |
| 7 | `--stage export` | Export + `--deploy` |

---

## Eight Models

| ID | Model | Recommended |
|----|-------|-------------|
| m1_tfidf_logreg | TF-IDF + Logistic Regression | Baseline |
| m2_tfidf_svm | TF-IDF + Linear SVM | Baseline |
| m3_roberta_base | RoBERTa-base | Production |
| **m4_roberta_focal** | **RoBERTa + Focal Loss** | **Default** |
| m5_distilroberta | DistilRoBERTa | Edge deploy |
| m6_deberta_v3 | DeBERTa-v3-base | Max accuracy |
| m7_xlm_roberta | XLM-RoBERTa-base | Multilingual |
| m8_captum_ig | Captum IG | XAI required |

```bash
python scripts/run_pipeline.py --model-id m4_roberta_focal --deploy
```

See [docs/IEEE_METHODOLOGY.md](docs/IEEE_METHODOLOGY.md) and [docs/02-eight-models.md](docs/02-eight-models.md).
