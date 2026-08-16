import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.explainability import compute_integrated_gradients
from app.labels import BASE_MODEL, DISPLAY_LABELS, ID2LABEL, LABEL2ID, MAX_LENGTH, NUM_LABELS

logger = logging.getLogger(__name__)


class EmotionInferenceService:
    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or os.getenv("MODEL_PATH", "./saved_emotion_model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.id2label = dict(ID2LABEL)
        self.label2id = dict(LABEL2ID)
        self._load()

    def _resolve_model_path(self) -> Path:
        path = Path(self.model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        return path

    def _load_label_map(self, model_dir: Path) -> None:
        label_map_file = model_dir / "label_map.json"
        if not label_map_file.exists():
            return
        with label_map_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.id2label = {int(k): str(v) for k, v in payload["id2label"].items()}
        self.label2id = {str(k): int(v) for k, v in payload["label2id"].items()}

    def _load(self) -> None:
        model_dir = self._resolve_model_path()
        self._load_label_map(model_dir)

        if (model_dir / "config.json").exists():
            logger.info("Loading fine-tuned model from %s", model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        else:
            logger.warning(
                "Fine-tuned weights not found at %s. Falling back to %s with 7-class head for development.",
                model_dir,
                BASE_MODEL,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL,
                num_labels=NUM_LABELS,
                id2label=self.id2label,
                label2id=self.label2id,
            )

        self.model.to(self.device)
        self.model.eval()

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _predict_logits(self, text: str) -> Tuple[int, str, float, Dict[str, float]]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        category = int(torch.argmax(probabilities).item())
        label = self.id2label[category]
        confidence = float(probabilities[category].item())
        scores = {
            self.id2label[idx]: float(probabilities[idx].item())
            for idx in range(len(probabilities))
        }
        return category, label, confidence, scores

    def predict(self, text: str) -> Dict[str, object]:
        category, label, confidence, scores = self._predict_logits(text)
        return {
            "category": category,
            "label": label,
            "display_label": DISPLAY_LABELS.get(category, label),
            "confidence": confidence,
            "scores": scores,
        }

    def explain(self, text: str, target_class: Optional[int] = None) -> Dict[str, object]:
        prediction = self.predict(text)
        category = target_class if target_class is not None else prediction["category"]
        tokens, heatmap = compute_integrated_gradients(
            model=self.model,
            tokenizer=self.tokenizer,
            text=text,
            target_class=category,
            device=self.device,
        )
        return {
            **prediction,
            "category": category,
            "label": self.id2label[category],
            "display_label": DISPLAY_LABELS.get(category, self.id2label[category]),
            "tokens": tokens,
            "heatmap": heatmap,
            "method": "integrated_gradients",
        }

    def chat(self, text: str) -> Dict[str, object]:
        explanation = self.explain(text)
        reply = self._build_reply(explanation["display_label"], explanation["confidence"])
        return {
            **explanation,
            "reply": reply,
        }

    @staticmethod
    def _build_reply(display_label: str, confidence: float) -> str:
        return (
            f"I classified this message as **{display_label}** "
            f"with {confidence:.1%} confidence. Review the token heatmap to see which words influenced the decision."
        )
