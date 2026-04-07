# 🛍️ FastAPI ML Backend — Smart Product Recommender

A production-ready **ML-powered product recommendation API** using collaborative filtering, built with FastAPI and deployed on GCP Cloud Run.

**Stack:** Python · FastAPI · scikit-learn · PostgreSQL · SQLAlchemy · Docker · GCP Cloud Run · GitHub Actions

---

## 🏗️ Architecture

```
Client Request
      │
      ▼
FastAPI (Port 8080)
      │
      ├── /api/v1/users        → User management (CRUD)
      ├── /api/v1/products     → Product catalogue (CRUD)
      └── /api/v1/recommend
              ├── /train       → Train ML model on interactions
              ├── /user        → Get personalised recommendations
              └── /model-info  → Model metadata

ML Engine (in-process):
  User-Item Matrix → Cosine Similarity → Top-K Recommendations
  Cold Start fallback → Popular Items

Storage:
  SQLite (local dev) / PostgreSQL (production GCP)
  Model persisted to models/recommender.pkl
```

---

## ✨ Features

- 🤖 **Collaborative filtering** — cosine similarity on user-item interaction matrix
- 🥶 **Cold start handling** — returns popular items for new/unknown users
- 💾 **Model persistence** — trained model saved to disk, reloaded on restart
- 📦 **Full CRUD** — users and products management endpoints
- 🧪 **8 test cases** — training, recommendation, cold start, CRUD
- 🐳 **Docker ready** — single `docker run` command
- ☁️ **GCP Cloud Run** — auto-scaled, serverless deployment
- 🔄 **CI/CD** — GitHub Actions auto-deploys on push to main

---

## 🚀 Quick Start

### Run locally
```bash
git clone https://github.com/Deepika230995/fastapi-recommender
cd fastapi-recommender
pip install -r requirements.txt
mkdir -p models
uvicorn app.main:app --reload --port 8080
```

Visit **http://localhost:8080/docs** for interactive Swagger UI.

### Run with Docker
```bash
docker build -t fastapi-recommender .
docker run -p 8080:8080 fastapi-recommender
```

### Seed with demo data
```bash
# Start the server first, then:
python scripts/seed_data.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + model status |
| POST | `/api/v1/users/` | Create user |
| GET | `/api/v1/users/{id}` | Get user |
| POST | `/api/v1/products/` | Create product |
| GET | `/api/v1/products/{id}` | Get product |
| POST | `/api/v1/recommend/train` | Train model on interactions |
| POST | `/api/v1/recommend/user` | Get recommendations for user |
| GET | `/api/v1/recommend/model-info` | Model metadata |

---

## 💡 Example Workflow

### Step 1 — Train the model
```bash
curl -X POST http://localhost:8080/api/v1/recommend/train \
  -H "Content-Type: application/json" \
  -d '{
    "interactions": [
      {"user_id": "u1", "product_id": "p1", "rating": 5.0},
      {"user_id": "u1", "product_id": "p2", "rating": 4.0},
      {"user_id": "u2", "product_id": "p1", "rating": 4.5},
      {"user_id": "u2", "product_id": "p3", "rating": 5.0},
      {"user_id": "u3", "product_id": "p2", "rating": 3.5},
      {"user_id": "u3", "product_id": "p3", "rating": 4.0}
    ]
  }'
```

```json
{
  "status": "trained",
  "details": {
    "users": 3,
    "products": 3,
    "interactions": 6,
    "trained_at": "2025-10-15T10:30:00"
  }
}
```

### Step 2 — Get Recommendations
```bash
curl -X POST http://localhost:8080/api/v1/recommend/user \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u1", "top_k": 3}'
```

```json
{
  "user_id": "u1",
  "recommendations": [
    {"rank": 1, "product_id": "p3", "score": 4.2341},
    {"rank": 2, "product_id": "p5", "score": 3.8812},
    {"rank": 3, "product_id": "p7", "score": 3.1204}
  ],
  "count": 3
}
```

---

## 🧪 Run Tests
```bash
pytest tests/ -v
```

Expected output:
```
tests/test_recommender.py::test_health PASSED
tests/test_recommender.py::test_train_model PASSED
tests/test_recommender.py::test_train_insufficient_data PASSED
tests/test_recommender.py::test_recommend_known_user PASSED
tests/test_recommender.py::test_recommend_cold_start_user PASSED
tests/test_recommender.py::test_create_and_get_user PASSED
tests/test_recommender.py::test_create_and_get_product PASSED
tests/test_recommender.py::test_model_info PASSED

8 passed in 1.24s
```

---

## 📁 Project Structure
```
fastapi-recommender/
├── app/
│   ├── main.py                    # FastAPI app + router registration
│   ├── api/
│   │   ├── health.py              # Health endpoint
│   │   ├── recommendations.py     # Train + recommend endpoints
│   │   ├── users.py               # User CRUD
│   │   └── products.py            # Product CRUD
│   ├── ml/
│   │   └── recommender.py         # Collaborative filtering engine
│   └── core/
│       └── database.py            # SQLAlchemy setup
├── tests/
│   └── test_recommender.py        # 8 test cases
├── scripts/
│   └── seed_data.py               # Demo data seeder
├── .github/workflows/
│   └── ci-cd.yml                  # GitHub Actions CI/CD → GCP Cloud Run
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ☁️ GCP Cloud Run Deployment

CI/CD auto-deploys on every push to `main`.

**Required GitHub Secrets:**
- `GCP_PROJECT_ID`
- `GCP_SA_KEY`
- `DATABASE_URL` (PostgreSQL connection string)

---

## 🔮 Future Improvements
- [ ] Matrix Factorization (SVD) for better accuracy
- [ ] Real-time interaction logging to PostgreSQL
- [ ] A/B testing framework for model versions
- [ ] Redis caching for hot recommendations
- [ ] Django admin panel for dataset management

---

## 👩‍💻 Author
**Lakshya Deepika P** — Backend & AI Engineer
- GitHub: [github.com/Deepika230995](https://github.com/Deepika230995)
- LinkedIn: [linkedin.com/in/lakshya-deepika](https://linkedin.com/in/lakshya-deepika)
- Email: lakshyadeepika042@gmail.com
"# fastapi-recommender" 
