from fastapi import APIRouter, Request

from app.models import Product
from app.work import request_work

router = APIRouter(tags=["products"])

PRODUCTS = (
    Product(id=1, name="Caderno", price=25.0, available=True),
    Product(id=2, name="Caneta", price=5.0, available=True),
    Product(id=3, name="Estojo", price=12.5, available=True),
)


@router.get(
    "/products",
    response_model=list[Product],
    responses={
        500: {"description": "Falha experimental"},
        503: {"description": "Indisponibilidade experimental ou do Go"},
    },
)
def products(request: Request) -> tuple[Product, ...]:
    request_work(request)
    return PRODUCTS
