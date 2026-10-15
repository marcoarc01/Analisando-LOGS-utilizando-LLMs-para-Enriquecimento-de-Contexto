from fastapi import APIRouter, HTTPException, Request, Response, status

from app.experiment import apply_experiment
from app.models import OrderRequest, OrderResponse
from app.routes.products import PRODUCTS

router = APIRouter(tags=["orders"])


@router.post(
    "/orders",
    response_model=OrderResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"description": "Produto não encontrado"},
        500: {"description": "Falha experimental"},
        503: {"description": "Indisponibilidade experimental ou do Go"},
    },
)
async def create_order(
    order: OrderRequest, request: Request
) -> OrderResponse | Response:
    product = next((item for item in PRODUCTS if item.id == order.product_id), None)
    if product is None:
        request.state.experiment.event = "product_not_found"
        raise HTTPException(status_code=404, detail="Produto não encontrado")

    override = await apply_experiment(request.app.state, request.state.experiment)
    if override is not None:
        return override

    return OrderResponse(
        product_id=product.id,
        quantity=order.quantity,
        unit_price=product.price,
        total=product.price * order.quantity,
    )
