import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import Settings


class StresserUnavailable(Exception):
    pass


class StressState(BaseModel):
    stress_level: int = Field(strict=True, ge=0, le=100)


class StresserClient:
    def __init__(
        self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None
    ):
        self.http = httpx.AsyncClient(
            base_url=str(settings.stresser_base_url).rstrip("/"),
            timeout=httpx.Timeout(settings.stresser_timeout_seconds),
            transport=transport or httpx.AsyncHTTPTransport(retries=0),
            follow_redirects=False,
            trust_env=False,
        )

    async def get_level(self) -> int:
        try:
            response = await self.http.get("/api/state")
            response.raise_for_status()
            return StressState.model_validate(response.json()).stress_level
        except (httpx.HTTPError, ValueError, ValidationError) as error:
            raise StresserUnavailable("Dependência indisponível") from error

    async def aclose(self) -> None:
        await self.http.aclose()
