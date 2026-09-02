# Kaggle Notebook Setup

Use these files to run training on Kaggle GPU.

## Option A: Upload notebook manually

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Settings: **GPU T4 x2**, **Internet ON**
3. Add data: **shivamb/go-emotions-google-emotions-dataset**
4. Upload or paste [`lab_final.ipynb`](../lab_final.ipynb)
5. Run all cells

## Option B: Push kernel from CLI

Install [Kaggle CLI](https://github.com/Kaggle/kaggle-api), configure `~/.kaggle/kaggle.json`, then:

```bash
cd notebooks/kaggle
kaggle kernels push -p .
```

This uses [`kernel-metadata.json`](kernel-metadata.json) which references `lab_final.ipynb` and the GoEmotions dataset.

## Option C: Run locally (CPU/GPU)

```bash
cd notebooks
pip install -r requirements-train.txt
python run_full_pipeline.py --deploy
```

## After training

Download `artifacts/exports/saved_emotion_model/` from Kaggle output, or use `--deploy` locally to copy into `packages/model/saved_emotion_model/`.
