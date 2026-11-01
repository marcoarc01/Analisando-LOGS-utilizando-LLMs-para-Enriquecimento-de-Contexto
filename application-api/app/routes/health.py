from fastapi import APIRouter, Request

from app.models import HealthResponse, LiveResponse
from app.work import request_work

router = APIRouter(tags=["health"])


@router.get("/live", response_model=LiveResponse)
def live() -> LiveResponse:
    return LiveResponse()


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        500: {"description": "Falha experimental"},
        503: {"description": "Indisponibilidade experimental ou do Go"},
    },
)
def health(request: Request) -> HealthResponse:
    request_work(request)
    return HealthResponse()


@router.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"description": "Go indisponível"}},
)
def ready() -> HealthResponse:
    return HealthResponse()
