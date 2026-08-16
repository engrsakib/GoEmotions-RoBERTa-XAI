import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.inference import EmotionInferenceService
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ExplainResponse,
    HealthResponse,
    PredictResponse,
    TextRequest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GoEmotions RoBERTa XAI Model Service",
    version="1.0.0",
    description="Fine-tuned RoBERTa inference and token-level Integrated Gradients explainability.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_service = EmotionInferenceService(model_path=os.getenv("MODEL_PATH", "./saved_emotion_model"))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if inference_service.is_ready else "degraded",
        model_loaded=inference_service.is_ready,
        model_path=str(inference_service._resolve_model_path()),
        device=str(inference_service.device),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: TextRequest) -> PredictResponse:
    try:
        result = inference_service.predict(payload.text.strip())
        return PredictResponse(**result)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/explain", response_model=ExplainResponse)
def explain(payload: TextRequest) -> ExplainResponse:
    try:
        result = inference_service.explain(payload.text.strip())
        return ExplainResponse(**result)
    except Exception as exc:
        logger.exception("Explainability failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result = inference_service.chat(payload.text.strip())
        return ChatResponse(**result)
    except Exception as exc:
        logger.exception("Chat classification failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
