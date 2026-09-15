# Related Work

This related-work summary is grounded in the local `notebooks/papers/` collection and organized into three categories that directly motivate the repository's GoEmotions-based workflow.

## Traditional ML/Lexicon Approaches

Early emotion detection relied on lexicons, manually engineered features, and classical classifiers. These methods remain important because they established the first practical baselines for affect classification. In the GoEmotions setting, the comparison in `6.pdf` shows that TF-IDF with logistic regression can still perform reasonably well on frequent emotions and some micro-level metrics [6]. However, the same study also shows that such models are much weaker on rare emotions and overlapping labels, so their role in this repository is mainly that of a baseline [6].

## BERT/RoBERTa-based Emotion Detection

Transformer-based emotion detection became dominant once contextual encoders showed clear gains over classical and recurrent methods. The most influential benchmark is GoEmotions, which introduced about 58,000 Reddit comments labeled with 27 emotions plus neutral and showed that even a BERT baseline reaches only about 0.46 average F1 on the original taxonomy [15]. This established GoEmotions as a genuinely difficult benchmark rather than a solved task.

Later work consistently strengthened the transformer line. The comparison in `6.pdf` reports that BERT-style models achieve the best overall balance on GoEmotions when compared with TF-IDF and BiLSTM baselines [6]. The study in `8.pdf` highlights RoBERTa as a strong alternative [8], while `16.pdf` also shows that contextual transformer models outperform SVM and BiLSTM for fine-grained emotion detection, even though confusion between closely related emotions remains common [16]. Some papers add extra structure through semantic graphs [2] or path-aware knowledge injection [1]. Together, these findings support the repository's choice to prioritize transformer encoders such as `m6_deberta_v3`.

## Multi-Label & Asymmetric Loss Innovations

The third category is the most directly related to this repository: multi-label modeling and imbalance-aware optimization. Fine-grained emotion data is naturally multi-label because one comment may express several emotions at the same time. Seq2Emo addresses this by modeling label correlation explicitly rather than treating each label as fully independent [4].

The biggest practical challenge, however, is imbalance. In GoEmotions, frequent labels dominate training while rare emotions receive much weaker supervision. The clipped asymmetric loss paper in `3.pdf` is the clearest direct precedent for this repository, showing that imbalance-aware optimization improves macro-F1 on GoEmotions from a BERT baseline of 0.46 to 0.54 [3]. Its key contribution is to reduce the influence of easy negatives while giving stronger focus to minority emotions [3]. This aligns closely with the repository's Track A setup, where `m6_deberta_v3` uses asymmetric loss and per-class threshold tuning.

Recent work also shows that calibration matters. The multilingual synthetic-data study emphasizes broader multi-label coverage [7], while the diffusion-based classifier paper shows that threshold selection can materially affect multi-label results [10]. Taken together, these studies motivate the repository's use of multi-hot labels, asymmetric loss, and validation-based threshold tuning [3], [4], [10], [15].

## Reference Note

- `[1]` `notebooks/papers/1.pdf` — *Path-Aware Knowledge Injection for Fine-Grained Emotion Recognition in Mental Health Counseling*
- `[2]` `notebooks/papers/2.pdf` — *Emotion Classification in Texts Over Graph Neural Networks: Semantic Representation is Better Than Syntactic*
- `[3]` `notebooks/papers/3.pdf` — *Improving Multi-Label Emotion Classification on Imbalanced Social Media Data With BERT and Clipped Asymmetric Loss*
- `[4]` `notebooks/papers/4.pdf` — *Seq2Emo: A Sequence to Multi-Label Emotion Classification Model*
- `[6]` `notebooks/papers/6.pdf` — *Fine-Grained Emotion Detection on GoEmotions: Experimental Comparison of Classical Machine Learning, BiLSTM, and Transformer Models*
- `[7]` `notebooks/papers/7.pdf` — *Multilingual Multi-Label Emotion Classification at Scale with Synthetic Data*
- `[8]` `notebooks/papers/8.pdf` — *Fine-Grained Classification for Emotion Detection Using Advanced Neural Models and GoEmotions Dataset*
- `[10]` `notebooks/papers/10.pdf` — *Discrete Diffusion Language Models Are Training-Free Multi-Label Classifiers*
- `[15]` `notebooks/papers/15.pdf` — *GoEmotions: A Dataset of Fine-Grained Emotions*
- `[16]` `notebooks/papers/16.pdf` — *Improving Fine-Grained Emotion Detection in Text with BERT and GoEmotions*
