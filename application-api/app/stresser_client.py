import logging
import time

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import Settings

# Closed list of httpcore trace points (per-request "trace" extension) that become
# dependency_* events. Every other trace point is ignored, and no trace payload
# (host, port, status, headers, timings) is ever copied into a log.
TRACE_EVENTS = {
    "connection.connect_tcp.complete": "dependency_connection_opened",
    "http11.send_request_body.complete": "dependency_request_sent",
    "http11.receive_response_headers.complete": "dependency_response_received",
}
# httpcore closes an HTTP/1.1 connection inside the request when the exchange did
# not finish (h11 states not DONE), i.e. after a failure. Its own "close" trace is
# emitted without a request, so it never reaches this callback.
CLOSED_AFTER_FAILURE = "http11.response_closed.complete"


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
        self.logger: logging.Logger | None = None

    def _emit(self, event: str, fields: dict, level: int = logging.INFO, **extra) -> None:
        if self.logger is not None:
            self.logger.log(level, event, extra={"event_fields": {**fields, **extra}})

    def _tracer(self, fields: dict, state: dict):
        async def trace(name: str, info: dict) -> None:
            if name.endswith(".failed"):
                if not state["failed"]:
                    state["failed"] = True
                    error = type(info.get("exception")).__name__
                    self._emit("dependency_response_failed", fields, logging.WARNING, error=error)
            elif name == CLOSED_AFTER_FAILURE:
                if state["failed"]:
                    self._emit("dependency_connection_closed", fields)
            elif name in TRACE_EVENTS:
                self._emit(TRACE_EVENTS[name], fields)

        return trace

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
        fields = {"request_id": request_id}
        if experiment_run_id:
            fields["experiment_run_id"] = experiment_run_id
        state = {"failed": False}
        try:
            started = time.perf_counter_ns()
            try:
                response = await self.http.get(
                    "/api/state", headers=headers,
                    extensions={"trace": self._tracer(fields, state)},
                )
            except httpx.HTTPError as error:
                self._record(started, type(error).__name__)
                if not state["failed"]:
                    # Failures without a trace point (e.g. PoolTimeout).
                    state["failed"] = True
                    self._emit("dependency_response_failed", fields, logging.WARNING,
                               error=type(error).__name__)
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
