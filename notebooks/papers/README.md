# Reference Papers (MIST Collection)

Bibliography for the GoEmotions-RoBERTa-XAI IEEE journal work. PDFs live in this folder.

| File | Title | Relevance |
|------|-------|-----------|
| [1.pdf](1.pdf) | Path-Aware Knowledge Injection for Fine-Grained Emotion Recognition in Mental Health Counseling | Knowledge injection, counseling domain |
| [2.pdf](2.pdf) | Emotion Classification in Texts Over Graph Neural Networks | GNN semantic representations |
| [3.pdf](3.pdf) | Improving Multi-Label Emotion Classification on Imbalanced Social Media Data With BERT and Clipped Asymmetric Loss | **Primary:** asymmetric loss, GoEmotions macro-F1 0.54 |
| [4.pdf](4.pdf) | Seq2Emo: A Sequence to Multi-Label Emotion Classification Model | Multi-label sequence modeling |
| [5.pdf](5.pdf) | (IEEE Access — emotion/NLP) | Related transformer work |
| [6.pdf](6.pdf) | Fine-Grained Emotion Detection on GoEmotions: Experimental Comparison | **Primary:** official split, BERT multi-label baseline |
| [7.pdf](7.pdf) | Multilingual Multi-Label Emotion Classification at Scale with Synthetic Data | Large-scale multi-label |
| [8.pdf](8.pdf) | Fine-Grained Classification (JSCDM 2024) | Fine-grained text classification |
| [9.pdf](9.pdf) | EmoS: A High-Fidelity Multimodal Benchmark (ACL 2026) | Multimodal emotion benchmark |
| [10.pdf](10.pdf) | Discrete Diffusion Language Models Are Training-Free Multi-Label Classifiers | Multi-label without fine-tuning |
| [11.pdf](11.pdf) | IEGPS-CSIC at SemEval-2025 Task 11: BERT-based Multi-label Emotion | SemEval multi-label BERT |
| [12.pdf](12.pdf) | Rethinking Emotion Annotations in the Era of Large Language Models | Annotation quality |
| [13.pdf](13.pdf) | Explicable Artificial Intelligence for Affective Computing (IEEE Intelligent Systems) | **Primary:** XAI faithfulness requirements |
| [14.pdf](14.pdf) | Multiclass Stress Detection Using Psychological Emotion Mapping and Transformer Architectures | Psychology mapping, hybrid BERT+RoBERTa |
| [15.pdf](15.pdf) | GoEmotions: A Dataset of Fine-Grained Emotions (Demszky et al., ACL 2020) | **Primary:** dataset paper |
| [16.pdf](16.pdf) | Improving Fine-Grained Emotion Detection in Text with BERT and GoEmotions | BERT + GoEmotions baseline |

## How This Repo Uses These Papers

- **Track A (IEEE):** Paper 3 asymmetric loss + Paper 6 official-split protocol on 7-macro multi-label heads
- **Track B (Production):** Paper 14 psychology mapping + distillation from Track A teacher
- **XAI:** Paper 13 faithfulness metrics (`src/xai/faithfulness.py`)
- **Dedup fix:** Avoid Paper 6-incompatible global dedup that drops 72% of rows with label conflicts

Run experiments: `python scripts/run_experiments.py --experiment E2`
