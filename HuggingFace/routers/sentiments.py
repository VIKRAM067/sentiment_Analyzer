from fastapi import APIRouter, HTTPException
from models import TextInput, BatchTextInput, sentimentResponse, BatchSentimentResponse
from ml_model import get_pipeline

router = APIRouter(prefix="/sentiment", tags=["Sentiment"])


@router.post("/analyze", response_model=sentimentResponse)
def analyze_sentiment(payload: TextInput):
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")
    
    pipe = get_pipeline()
    result = pipe(payload.text)[0]
    
    return sentimentResponse(
        text=payload.text,
        label=result["label"],
        score=round(result["score"], 4)
    )


@router.post("/batch", response_model=BatchSentimentResponse)
def analyze_batch(payload: BatchTextInput):
    if not payload.texts:
        raise HTTPException(status_code=422, detail="Texts list cannot be empty")
    if len(payload.texts) > 32:
        raise HTTPException(status_code=400, detail="Max 32 texts per batch")
    
    pipe = get_pipeline()
    results = pipe(payload.texts)
    
    items = [
        sentimentResponse(text=text, label=r["label"], score=round(r["score"], 4))
        for text, r in zip(payload.texts, results)
    ]
    return BatchSentimentResponse(results=items, count=len(items))


@router.get("/model-info")
def model_info():
    return {
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": "text-classification",
        "labels": ["POSITIVE", "NEGATIVE"],
        "max_tokens": 512,
        "source": "HuggingFace"
    }