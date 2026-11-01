import time

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import Settings


class StresserUnavailable(Exception):
    def __init__(self, error_type: str):
        super().__init__("Dependência indisponível")
        # Only the error class name or a fixed category: never the exception
        # text, which may contain URLs, hosts or ports of the dependency.
        self.error_type = error_type


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
        self.timing = (
            open(settings.lookup_timing_file, "a", encoding="ascii", buffering=1)
            if settings.lookup_timing_file
            else None
        )

    def _record(self, started_ns: int, outcome: str) -> None:
        if self.timing is not None:
            ended = time.perf_counter_ns()
            self.timing.write(f"{time.time_ns()},{ended - started_ns},{outcome}\n")

    async def get_level(
        self, *, request_id: str | None = None, experiment_run_id: str | None = None
    ) -> int:
        headers = {}
        if request_id:
            headers["X-Request-ID"] = request_id
        if experiment_run_id:
            headers["X-Experiment-Run-ID"] = experiment_run_id
        try:
            started = time.perf_counter_ns()
            try:
                response = await self.http.get("/api/state", headers=headers)
            except httpx.HTTPError as error:
                self._record(started, type(error).__name__)
                raise
            self._record(started, "ok")
            response.raise_for_status()
            return StressState.model_validate(response.json()).stress_level
        except httpx.HTTPStatusError as error:
            raise StresserUnavailable("UnexpectedStatus") from error
        except httpx.HTTPError as error:
            raise StresserUnavailable(type(error).__name__) from error
        except (ValueError, ValidationError) as error:
            raise StresserUnavailable("InvalidResponse") from error

    async def aclose(self) -> None:
        await self.http.aclose()
        if self.timing is not None:
            self.timing.close()
