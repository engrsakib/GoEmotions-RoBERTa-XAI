# 04 — Export Checklist

After training, export the best checkpoint to production.

## Export Location

```
notebooks/artifacts/exports/saved_emotion_model/
```

Copy locally to:

```
packages/model/saved_emotion_model/
```

## Required Files

- [ ] `config.json` — model architecture + num_labels=7
- [ ] `model.safetensors` or `pytorch_model.bin` — weights
- [ ] `tokenizer.json` — fast tokenizer
- [ ] `vocab.json` — RoBERTa vocabulary
- [ ] `merges.txt` — BPE merges
- [ ] `special_tokens_map.json` — special tokens
- [ ] `label_map.json` — id2label / label2id (7 classes)

## Label Map Verification

`label_map.json` must match production:

```json
{
  "id2label": {
    "0": "neutral",
    "1": "sadness_grief",
    "2": "joy_amusement_excitement_optimism",
    "3": "anger_annoyance_disapproval_disgust",
    "4": "desire",
    "5": "fear_nervousness",
    "6": "love"
  }
}
```

## Export Code

```python
from src.training.export import export_model

export_model(trainer, tokenizer, export_dir="artifacts/exports/saved_emotion_model")
```

## Post-Export Smoke Test

```bash
cd packages/model
pip install -r requirements.txt
python -c "
from app.inference import EmotionInferenceService
svc = EmotionInferenceService(model_path='./saved_emotion_model')
r = svc.predict('I love this so much!')
print(r)
"
```

Expected: loads fine-tuned weights (no fallback warning), returns category 6 (love).

## XAI Smoke Test

```python
from app.inference import EmotionInferenceService
svc = EmotionInferenceService(model_path='./saved_emotion_model')
r = svc.explain('I am afraid of the dark.')
assert len(r['tokens']) == len(r['heatmap'])
print('XAI OK:', r['display_label'])
```

## Kaggle Dataset Publish

1. Zip `artifacts/exports/saved_emotion_model/`
2. Create new Kaggle dataset
3. Reference in Docker / CI as needed

## Success Criteria

- [ ] macro-F1 on test set beats TF-IDF baseline
- [ ] per-class F1 for sadness & desire > 0.35
- [ ] `EmotionInferenceService` loads without fallback
- [ ] Captum heatmaps render for sample texts
