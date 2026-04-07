"""
Collaborative Filtering Recommendation Engine
Uses cosine similarity on user-item interaction matrix.
Author: Lakshya Deepika P
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

MODEL_PATH = "models/recommender.pkl"


class RecommendationEngine:
    """
    User-based collaborative filtering recommender.
    Finds similar users and recommends products they liked.
    """

    def __init__(self):
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.trained_at = None
        self.is_trained = False

    def train(self, interactions: list[dict]) -> dict:
        """
        Train on user-item interactions.
        interactions: [{"user_id": 1, "product_id": 10, "rating": 4.5}, ...]
        """
        if not interactions:
            raise ValueError("Need at least some interactions to train")

        df = pd.DataFrame(interactions)

        # Encode IDs
        df["user_idx"] = self.user_encoder.fit_transform(df["user_id"].astype(str))
        df["item_idx"] = self.item_encoder.fit_transform(df["product_id"].astype(str))

        # Build user-item matrix
        n_users = df["user_idx"].nunique()
        n_items = df["item_idx"].nunique()
        matrix = np.zeros((n_users, n_items))

        for _, row in df.iterrows():
            matrix[int(row["user_idx"]), int(row["item_idx"])] = row["rating"]

        self.user_item_matrix = matrix
        self.similarity_matrix = cosine_similarity(matrix)
        self.trained_at = datetime.utcnow().isoformat()
        self.is_trained = True

        os.makedirs("models", exist_ok=True)
        joblib.dump(self, MODEL_PATH)
        logger.info(f"Model trained: {n_users} users, {n_items} products")

        return {
            "users": n_users,
            "products": n_items,
            "interactions": len(df),
            "trained_at": self.trained_at
        }

    def recommend(self, user_id: str, top_k: int = 5) -> list[dict]:
        """Recommend top_k products for a given user."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call /model/train first.")

        try:
            user_idx = self.user_encoder.transform([str(user_id)])[0]
        except ValueError:
            # Cold start — return top-rated products across all users
            logger.info(f"Cold start for user {user_id}, returning popular items")
            return self._popular_items(top_k)

        # Find similar users
        similarities = self.similarity_matrix[user_idx]
        similar_users = np.argsort(similarities)[::-1][1:6]  # top 5 similar users

        # Aggregate ratings from similar users for unseen products
        user_ratings = self.user_item_matrix[user_idx]
        seen_items = set(np.where(user_ratings > 0)[0])

        scores = np.zeros(self.user_item_matrix.shape[1])
        for sim_user_idx in similar_users:
            sim = similarities[sim_user_idx]
            scores += sim * self.user_item_matrix[sim_user_idx]

        # Exclude already seen items
        for idx in seen_items:
            scores[idx] = 0

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                try:
                    product_id = self.item_encoder.inverse_transform([idx])[0]
                    results.append({
                        "rank": rank + 1,
                        "product_id": product_id,
                        "score": round(float(scores[idx]), 4)
                    })
                except Exception:
                    continue

        return results if results else self._popular_items(top_k)

    def _popular_items(self, top_k: int) -> list[dict]:
        """Fallback: return most interacted-with items."""
        if self.user_item_matrix is None:
            return []
        item_scores = self.user_item_matrix.sum(axis=0)
        top_indices = np.argsort(item_scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            try:
                product_id = self.item_encoder.inverse_transform([idx])[0]
                results.append({
                    "rank": rank + 1,
                    "product_id": product_id,
                    "score": round(float(item_scores[idx]), 4),
                    "note": "popular_fallback"
                })
            except Exception:
                continue
        return results


def load_engine() -> RecommendationEngine:
    """Load trained model or return fresh instance."""
    if os.path.exists(MODEL_PATH):
        logger.info("Loading recommendation engine from disk")
        return joblib.load(MODEL_PATH)
    logger.info("No trained model found, starting fresh")
    return RecommendationEngine()


# Singleton instance
engine = load_engine()
