from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from ml_model import get_pipeline
from routers import sentiments

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()   # pre-warm model at startup
    yield

app = FastAPI(
    title="Sentiment Analysis API",
    description="HuggingFace DistilBERT powered sentiment analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Sentiment API is running"}

app.include_router(sentiments.router)