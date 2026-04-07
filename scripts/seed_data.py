"""
Seed script — populates the API with demo users, products, and trains the model.
Run: python scripts/seed_data.py
"""

import requests
import random

BASE = "http://localhost:8080"

USERS = [
    {"user_id": f"u{i}", "name": f"User {i}", "email": f"user{i}@example.com"}
    for i in range(1, 21)
]

PRODUCTS = [
    {"product_id": f"p{i}", "name": f"Product {i}",
     "category": random.choice(["Electronics", "Books", "Clothing", "Food"]),
     "price": round(random.uniform(9.99, 199.99), 2)}
    for i in range(1, 31)
]

INTERACTIONS = [
    {"user_id": f"u{random.randint(1,20)}",
     "product_id": f"p{random.randint(1,30)}",
     "rating": round(random.uniform(1.0, 5.0), 1)}
    for _ in range(200)
]


def seed():
    print("Creating users...")
    for user in USERS:
        requests.post(f"{BASE}/api/v1/users/", json=user)

    print("Creating products...")
    for product in PRODUCTS:
        requests.post(f"{BASE}/api/v1/products/", json=product)

    print("Training recommendation model...")
    r = requests.post(f"{BASE}/api/v1/recommend/train",
                      json={"interactions": INTERACTIONS})
    print(f"Training result: {r.json()}")

    print("\nTesting recommendations for user u1:")
    r = requests.post(f"{BASE}/api/v1/recommend/user",
                      json={"user_id": "u1", "top_k": 5})
    for rec in r.json()["recommendations"]:
        print(f"  Rank {rec['rank']}: Product {rec['product_id']} (score: {rec['score']})")

    print("\n✅ Seed complete!")


if __name__ == "__main__":
    seed()
