"""Products router"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_products: dict[str, dict] = {}


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    description: str = ""


@router.post("/")
def create_product(product: Product):
    _products[product.product_id] = product.dict()
    return {"message": "Product created", "product": product}


@router.get("/{product_id}")
def get_product(product_id: str):
    if product_id not in _products:
        raise HTTPException(status_code=404, detail="Product not found")
    return _products[product_id]


@router.get("/")
def list_products():
    return {"products": list(_products.values()), "count": len(_products)}
