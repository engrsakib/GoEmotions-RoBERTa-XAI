# Kaggle Training Setup

## Option A: Run script (recommended)

1. Kaggle Notebook → **GPU T4 x2**, **Internet ON**
2. Add dataset: `shivamb/go-emotions-google-emotions-dataset`
3. Run:

```python
!git clone https://github.com/engrsakib/GoEmotions-RoBERTa-XAI.git /kaggle/working/repo
%cd /kaggle/working/repo/notebooks
!pip install -q -r requirements-train.txt
!python kaggle/run_training.py
```

## Option B: Kaggle CLI push

```bash
cd notebooks/kaggle
kaggle kernels push -p .
```

Uses [`kernel-metadata.json`](kernel-metadata.json) pointing to [`run_training.py`](run_training.py).

## Option C: Full pipeline CLI

```bash
python scripts/run_pipeline.py --deploy --model-id m4_roberta_focal
```

## Output

Download `artifacts/exports/saved_emotion_model/` after training completes.
