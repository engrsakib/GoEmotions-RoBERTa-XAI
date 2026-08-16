from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="GoEmotions RoBERTa XAI Model Service")


class TextIn(BaseModel):
    text: str


class Prediction(BaseModel):
    category: int
    label: str
    scores: Dict[str, float]
    tokens: List[str]
    heatmap: List[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(payload: TextIn):
    # Placeholder prediction: neutral
    tokens = payload.text.split()
    n = len(tokens) or 1
    heatmap = [0.0 for _ in range(n)]
    return Prediction(category=0, label="neutral", scores={"neutral": 1.0}, tokens=tokens, heatmap=heatmap)
