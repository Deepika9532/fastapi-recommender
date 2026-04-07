"""Recommendations API endpoints"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ml.recommender import engine
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class TrainRequest(BaseModel):
    interactions: list[dict]   # [{"user_id": 1, "product_id": 10, "rating": 4.5}]


class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = 5


@router.post("/train")
def train_model(req: TrainRequest):
    """Train the recommendation model on user-item interactions."""
    if len(req.interactions) < 5:
        raise HTTPException(
            status_code=400,
            detail="Need at least 5 interactions to train"
        )
    try:
        result = engine.train(req.interactions)
        return {"status": "trained", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/user")
def get_recommendations(req: RecommendRequest):
    """Get top-K product recommendations for a user."""
    if req.top_k < 1 or req.top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")

    try:
        recommendations = engine.recommend(str(req.user_id), req.top_k)
        return {
            "user_id": req.user_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
def model_info():
    """Get current model status and metadata."""
    return {
        "is_trained": engine.is_trained,
        "trained_at": engine.trained_at,
        "model_type": "User-Based Collaborative Filtering (Cosine Similarity)",
        "model_path": "models/recommender.pkl"
    }
