"""Health check router"""
from fastapi import APIRouter
from app.ml.recommender import engine

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "fastapi-recommender",
        "version": "1.0.0",
        "model_trained": engine.is_trained
    }
