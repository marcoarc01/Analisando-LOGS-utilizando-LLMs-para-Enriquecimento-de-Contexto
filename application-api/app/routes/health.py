from fastapi import APIRouter

from app.models import HealthResponse, LiveResponse

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
def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"description": "Go indisponível"}},
)
def ready() -> HealthResponse:
    return HealthResponse()
