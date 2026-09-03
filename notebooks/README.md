# GoEmotions-RoBERTa Training Pipeline

Professional Python pipeline (replaces `lab_final.ipynb`). Production inference lives in `packages/model/`.

## Quick Start (Local)

```bash
cd notebooks
pip install -r requirements-train.txt
python scripts/run_pipeline.py --deploy
```

## Quick Start (Kaggle GPU)

1. Enable **GPU T4 x2** and **Internet**
2. Add dataset: `shivamb/go-emotions-google-emotions-dataset`
3. Clone repo and run:

```bash
git clone https://github.com/engrsakib/GoEmotions-RoBERTa-XAI.git /kaggle/working/repo
cd /kaggle/working/repo/notebooks
pip install -r requirements-train.txt
python kaggle/run_training.py
```

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
│   ├── IEEE_METHODOLOGY.md       # IEEE-format methodology
│   └── 02-eight-models.md        # Eight recommended models
├── kaggle/
│   ├── run_training.py           # Kaggle entry point
│   └── kernel-metadata.json
└── artifacts/                    # gitignored outputs
```

## Pipeline Stages

| Stage | Script flag | Description |
|-------|-------------|-------------|
| 0 | (auto) | Bootstrap environment |
| 1 | `--stage data` | Data engineering |
| 2 | `--stage eda` | Class distribution figure |
| 3 | `--stage baselines` | M1 LogReg + M2 SVM |
| 4–5 | `--stage train` | Transformer fine-tune + evaluate |
| 6 | `--stage xai` | Captum heatmaps (M8) |
| 7 | `--stage export` | Export + optional `--deploy` |

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
# Train a specific transformer
python scripts/run_pipeline.py --model-id m6_deberta_v3 --skip-data --deploy
```

See [docs/IEEE_METHODOLOGY.md](docs/IEEE_METHODOLOGY.md) and [docs/02-eight-models.md](docs/02-eight-models.md).
