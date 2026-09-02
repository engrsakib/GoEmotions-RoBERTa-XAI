# GoEmotions-RoBERTa Training

All model training runs from this directory. Production inference lives in
`packages/model/`.

## Quick Start (Kaggle)

1. Create a Kaggle notebook with **GPU** and **Internet** enabled.
2. Add dataset: `shivamb/go-emotions-google-emotions-dataset`.
3. Open [`lab_final.ipynb`](lab_final.ipynb) and run all cells.

Or clone the repo manually:

```python
!git clone https://github.com/engrsakib/GoEmotions-RoBERTa-XAI-Fine-Tuned-Sentiment-Classifier-with-Token-Level-Attribution-Heatmaps.git /kaggle/working/repo
%cd /kaggle/working/repo/notebooks
!pip install -q -r requirements-train.txt
```

## Quick Start (Local)

```bash
cd notebooks
pip install -r requirements-train.txt
# Place GoEmotions CSV in data/raw/goemotions/ OR rely on kagglehub
jupyter notebook lab_final.ipynb
```

## Directory Layout

```
notebooks/
├── lab_final.ipynb          # Main linear pipeline
├── requirements-train.txt
├── config/train_config.yaml
├── docs/                    # Training documentation (.md)
├── src/                     # Importable Python modules
└── artifacts/               # Outputs (gitignored)
    ├── processed/
    ├── checkpoints/
    ├── logs/
    └── exports/
```

## Pipeline Stages

1. **Data engineering** — load, map 28→7 labels, clean, split, export
2. **Baseline** — TF-IDF + Logistic Regression (macro-F1 floor)
3. **RoBERTa fine-tuning** — Focal Loss + AdamW cosine schedule
4. **XAI validation** — Captum Integrated Gradients
5. **Export** — copy to `packages/model/saved_emotion_model/`

See [`docs/`](docs/) for detailed guides.
