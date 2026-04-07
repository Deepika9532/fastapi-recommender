"""Users router"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# In-memory store for demo (replace with PostgreSQL via SQLAlchemy in production)
_users: dict[str, dict] = {}


class User(BaseModel):
    user_id: str
    name: str
    email: str


@router.post("/")
def create_user(user: User):
    _users[user.user_id] = user.dict()
    return {"message": "User created", "user": user}


@router.get("/{user_id}")
def get_user(user_id: str):
    if user_id not in _users:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="User not found")
    return _users[user_id]


@router.get("/")
def list_users():
    return {"users": list(_users.values()), "count": len(_users)}
