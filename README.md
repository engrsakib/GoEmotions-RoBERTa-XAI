# 🚀 GoEmotions-RoBERTa-XAI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green?logo=fastapi&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch&logoColor=white) ![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-ff9900?logo=huggingface&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=nextdotjs&logoColor=white) ![React](https://img.shields.io/badge/React-18.2-blue?logo=react&logoColor=white) ![Node.js](https://img.shields.io/badge/Node.js-18-green?logo=node.js&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-blue?logo=docker&logoColor=white) ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI-2088ff?logo=githubactions&logoColor=white) ![ESLint](https://img.shields.io/badge/ESLint-Linting-4B32C3?logo=eslint&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-styling-06B6D4?logo=tailwindcss&logoColor=white) ![Captum](https://img.shields.io/badge/Captum-XAI-563D7C) ![ONNX](https://img.shields.io/badge/ONNX-Export-000000?logo=onnx&logoColor=white)

Project: Fine-Tuned Sentiment Classifier with Token-Level Attribution Heatmaps  
This repository provides a production-grade monorepo layout for a sentiment classification system based on RoBERTa fine-tuned on the GoEmotions dataset, augmented with token-level XAI heatmaps for interpretability. The stack includes a model service (Python/FastAPI), a Next.js (v16) frontend UI, and a Next.js (v16) backend API — all runnable from a single Docker image.

## Key Features
- RoBERTa fine-tuned classifier (GoEmotions)
- Token-level attribution heatmaps (Integrated Gradients / XAI)
- Monorepo: frontend, backend, and model service
- Single production Dockerfile to build and run all services
- GitHub Actions CI: functional checks, ESLint, and Docker build
- Grok-compatible API endpoint and a simple chatbot categorizer

## Classification Categories (chatbot mapping)
When a message is received, categorize into:
0: Normal (neutral)  
1: Sadness (sadness, grief)  
2: Joy (joy, amusement, excitement, optimism)  
3: Hate / Anger (anger, annoyance, disapproval, disgust)  
4: Sexual / Desire (desire)  
5: Fear / Anxiety (fear, nervousness)  
6: Love / ভালোবাসা (love)

## Technologies & Tools
- Python (3.10+), FastAPI, Uvicorn
- PyTorch / Transformers (Hugging Face)
- Captum / Integrated Gradients (or custom attribution)
- Node.js (18+/20+), Next.js 16 (App Router)
- React, Tailwind CSS (optional)
- Docker (single multi-stage image)
- GitHub Actions (CI)
- ESLint + Prettier
- PM2 / Supervisord (process management inside container)
- pytest / tox (testing)
- Weights & Biases or TensorBoard (experiment tracking)
- Numpy / Pandas / Scikit-learn
- ONNX (optional export for faster inference)
- Redis (optional caching / rate limiting)

12 Algorithms & Methods (you can use any subset during experimentation)
1. RoBERTa (Transformer-based encoder)  
2. Logistic Regression (baseline)  
3. SVM (baseline)  
4. LSTM (alternate deep model)  
5. CNN for text (alternate)  
6. DistilRoBERTa (compact transformer)  
7. Fine-tuning with AdamW (optimizer recipe)  
8. Focal Loss / Label smoothing (class handling)  
9. Integrated Gradients (attribution)  
10. Layer-wise Relevance Propagation (LRP) (optional)  
11. Grad-CAM for token-level (adaptation)  
12. ONNX model quantization / pruning (deployment optimizations)

## Dataset
- GoEmotions (Google) — multi-label emotion dataset.
- URL: https://github.com/google-research/google-research/tree/master/goemotions
- Place raw dataset under: `data/raw/goemotions/`

## Recommended Folder Structure
```
.
├── README.md
├── Dockerfile
├── .github/
│   └── workflows/ci.yml
├── .eslintrc.json
├── .gitignore
├── data/
│   └── raw/
├── packages/
│   ├── frontend/        # Next.js v16 UI
│   └── backend/         # Next.js v16 server-side API (app router)
├── services/
│   └── model/           # Python model service (FastAPI)
├── notebooks/           # research notebooks (includes lab_final.ipynb)
├── scripts/             # utility scripts: training, preprocessing, export
└── project-plan/        # (ignored) production plan & checklist
```

## Kaggle / Training Guidelines (high-level)
1. Prepare data: download GoEmotions, split into train/val/test, and map labels to your 7-class mapping (or use original multi-label and convert).  
2. Tokenization: use `roberta-base` tokenizer with MAX_LENGTH = 128 (adjust after analyzing token length distribution).  
3. Baselines: train Logistic Regression and SVM on bag-of-words / TF-IDF for quick baselines.  
4. Fine-tune RoBERTa: use Hugging Face Trainer or a custom PyTorch loop. Recommended hyperparams: batch_size 16, lr 2e-5, epochs 3-5, weight_decay 0.01. Use gradient accumulation if needed.  
5. Validation: track per-class F1 and macro F1. Early stop on macro F1.  
6. Explainability: compute token-level attributions with Integrated Gradients (captum) and normalize heatmaps for UI overlay.  
7. Export: save best model checkpoint and optionally export to ONNX for inference speedups.  
8. Reproducibility: seed everything, log experiments to W&B or TensorBoard, include environment.yml / requirements.txt.

## Docker & Production Run (single-image guideline)
- The root `Dockerfile` builds Python and Node artifacts, then runs a small process manager that starts:
  - `uvicorn services.model.app:app --host 0.0.0.0 --port 8000` (model API)
  - `next start -p 3000` for frontend
  - `next start -p 3001` for backend (if separate)
- Expose ports 8000 (model), 3000 (UI). Map them through Docker run flags or Nginx reverse proxy in the container.  
- Use environment variables to configure MODEL_PATH, GPU usage, and NODE_ENV=production.  
- Healthchecks: add `/health` endpoints on every service and configure Docker HEALTHCHECK.  
- Logging: JSON structured logs, rotate logs, and forward to STDOUT for container runtime.

## Grok API & Chatbot
- Provide an endpoint `/api/grok/classify` (POST) that accepts JSON: `{ "text": "..." }` and returns `{ "category": 3, "label": "anger", "scores": {...}, "tokens": [...], "heatmap": [...] }`.  
- A lightweight chatbot endpoint (`/api/chat`) should call the classifier and reply with the category and a short natural response.

## Next Steps I Can Do (tell me which first)
1. Create the skeleton files and CI workflow (eslint + tests + docker build).  
2. Add the multi-stage Dockerfile that builds both Node and Python artifacts.  
3. Scaffold minimal Next.js apps and a FastAPI model service with example endpoints.  
4. Wire the classifier to use the notebook training artifacts and provide model loading.  

Tell me which of the actions above you want me to implement now, or I can proceed to scaffold everything recommended in this README.

