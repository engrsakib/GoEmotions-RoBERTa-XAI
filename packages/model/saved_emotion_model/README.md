# Saved Emotion Model — Weight Drop Zone

Place your fine-tuned Hugging Face checkpoint here after training (`notebooks/lab_final.ipynb`).

## Required layout

```
saved_emotion_model/
├── label_map.json              # included (7-class mapping)
├── config.json                 # from trainer.save_model()
├── model.safetensors           # or pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
└── weights/                    # optional extra checkpoints / archives
```

## Copy from notebook

```python
SAVE_DIR = "./saved_emotion_model"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

Then copy the exported folder into:

`packages/model/saved_emotion_model/`

## Docker volume

`docker-compose.yml` mounts this directory read-only into the container at `/app/saved_emotion_model`.

## Fallback behavior

If `config.json` is missing, the FastAPI service loads `roberta-base` with a 7-class head for local development only. Replace with real weights before production.
