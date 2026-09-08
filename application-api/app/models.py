from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class LiveResponse(HealthResponse):
    service: Literal["application-api"] = "application-api"


class Product(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    price: float
    available: bool


class OrderRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    product_id: int
    quantity: int = Field(gt=0)


class OrderResponse(BaseModel):
    product_id: int
    quantity: int
    unit_price: float
    total: float
    status: Literal["simulated"] = "simulated"
