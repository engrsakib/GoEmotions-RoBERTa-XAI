from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class PredictResponse(BaseModel):
    category: int
    label: str
    display_label: str
    confidence: float
    scores: Dict[str, float]


class ExplainResponse(PredictResponse):
    tokens: List[str]
    heatmap: List[float]
    method: str = "integrated_gradients"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    category: int
    label: str
    display_label: str
    confidence: float
    reply: str
    scores: Dict[str, float]
    tokens: Optional[List[str]] = None
    heatmap: Optional[List[float]] = None
