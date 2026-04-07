"""Tests for FastAPI Recommender"""
import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.main import app

client = TestClient(app)

SAMPLE_INTERACTIONS = [
    {"user_id": "u1", "product_id": "p1", "rating": 5.0},
    {"user_id": "u1", "product_id": "p2", "rating": 4.0},
    {"user_id": "u1", "product_id": "p3", "rating": 3.0},
    {"user_id": "u2", "product_id": "p1", "rating": 4.5},
    {"user_id": "u2", "product_id": "p4", "rating": 5.0},
    {"user_id": "u2", "product_id": "p5", "rating": 4.0},
    {"user_id": "u3", "product_id": "p2", "rating": 3.5},
    {"user_id": "u3", "product_id": "p5", "rating": 4.5},
    {"user_id": "u3", "product_id": "p6", "rating": 5.0},
    {"user_id": "u4", "product_id": "p3", "rating": 2.0},
    {"user_id": "u4", "product_id": "p6", "rating": 4.0},
]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_train_model():
    response = client.post(
        "/api/v1/recommend/train",
        json={"interactions": SAMPLE_INTERACTIONS}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "trained"
    assert data["details"]["users"] > 0


def test_train_insufficient_data():
    response = client.post(
        "/api/v1/recommend/train",
        json={"interactions": [{"user_id": "u1", "product_id": "p1", "rating": 5}]}
    )
    assert response.status_code == 400


def test_recommend_known_user():
    # Train first
    client.post("/api/v1/recommend/train", json={"interactions": SAMPLE_INTERACTIONS})
    response = client.post(
        "/api/v1/recommend/user",
        json={"user_id": "u1", "top_k": 3}
    )
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 3


def test_recommend_cold_start_user():
    client.post("/api/v1/recommend/train", json={"interactions": SAMPLE_INTERACTIONS})
    response = client.post(
        "/api/v1/recommend/user",
        json={"user_id": "unknown_user_999", "top_k": 3}
    )
    assert response.status_code == 200
    # Cold start returns popular items
    assert "recommendations" in response.json()


def test_create_and_get_user():
    response = client.post("/api/v1/users/", json={
        "user_id": "test_u1",
        "name": "Lakshya Deepika",
        "email": "test@example.com"
    })
    assert response.status_code == 200
    response = client.get("/api/v1/users/test_u1")
    assert response.status_code == 200
    assert response.json()["name"] == "Lakshya Deepika"


def test_create_and_get_product():
    response = client.post("/api/v1/products/", json={
        "product_id": "prod_001",
        "name": "Test Product",
        "category": "Electronics",
        "price": 29.99
    })
    assert response.status_code == 200
    response = client.get("/api/v1/products/prod_001")
    assert response.status_code == 200
    assert response.json()["name"] == "Test Product"


def test_model_info():
    response = client.get("/api/v1/recommend/model-info")
    assert response.status_code == 200
    assert "is_trained" in response.json()
