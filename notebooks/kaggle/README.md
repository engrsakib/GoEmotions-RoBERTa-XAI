# Kaggle Training Setup

**Full guide:** [docs/03-kaggle-setup.md](../docs/03-kaggle-setup.md)

## Quick commands (main branch)

```python
# Clone + install + train (paste in Kaggle notebook)
import os, subprocess, sys

REPO = "https://github.com/engrsakib/GoEmotions-RoBERTa-XAI.git"
REPO_DIR = "/kaggle/working/repo"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--branch", "main", "--depth", "1", REPO, REPO_DIR], check=True)

os.chdir(f"{REPO_DIR}/notebooks")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-train.txt"], check=True)
subprocess.run([sys.executable, "kaggle/run_training.py"], check=True)
```

## Dataset

Add as Kaggle Input: [GoEmotions Google Emotions Dataset](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset)

## Kaggle CLI push (optional)

```bash
cd notebooks/kaggle
kaggle kernels push -p .
```
