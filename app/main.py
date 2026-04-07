"""
FastAPI ML Backend — Smart Product Recommender
Stack: Python · FastAPI · scikit-learn · PostgreSQL · Docker · GCP Cloud Run
Author: Lakshya Deepika P
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import recommendations, users, products, health
from app.core.database import engine, Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create DB tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart Product Recommender API",
    description="""
    ML-powered product recommendation service using collaborative filtering.
    Built with FastAPI + scikit-learn + PostgreSQL.
    Deployed on GCP Cloud Run with Docker.
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router,           tags=["Health"])
app.include_router(users.router,            prefix="/api/v1/users",   tags=["Users"])
app.include_router(products.router,         prefix="/api/v1/products", tags=["Products"])
app.include_router(recommendations.router,  prefix="/api/v1/recommend", tags=["Recommendations"])
