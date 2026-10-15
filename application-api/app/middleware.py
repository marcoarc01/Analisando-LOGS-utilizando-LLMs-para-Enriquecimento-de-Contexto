import logging
import re
from time import perf_counter
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.experiment import ExperimentContext, apply_experiment

EXPERIMENTAL_ROUTES = {
    ("GET", "/products"): "products_completed",
    ("POST", "/orders"): "order_created",
    ("GET", "/health"): "health_checked",
}
# /live and /ready are operational probes, excluded from experimental logs.
# /metrics access logging is deliberately disabled to avoid scrape noise.
KNOWN_ROUTES = {
    "/products",
    "/orders",
    "/health",
    "/live",
    "/ready",
    "/metrics",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
}
METHODS = {
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "HEAD",
    "OPTIONS",
    "TRACE",
    "CONNECT",
}


class ExperimentMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        state = scope["app"].state
        metrics = state.metrics
        started = perf_counter()
        method = scope["method"] if scope["method"] in METHODS else "OTHER"
        route = scope["path"] if scope["path"] in KNOWN_ROUTES else "__unmatched__"
        experimental = (method, route) in EXPERIMENTAL_ROUTES
        needs_lookup = (experimental and (method, route) != ("POST", "/orders")) or (
            method,
            route,
        ) == ("GET", "/ready")
        headers = Headers(scope=scope)
        request_id = headers.get("X-Request-ID", "")
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id):
            request_id = str(uuid4())
        run_id = headers.get("X-Experiment-Run-ID", "")
        fields = {"method": method, "route": route, "request_id": request_id}
        if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", run_id):
            fields["experiment_run_id"] = run_id

        context = ExperimentContext(
            request_id=request_id,
            experiment_run_id=fields.get("experiment_run_id"),
        )
        scope.setdefault("state", {})["experiment"] = context
        status_code = 500
        override = None

        async def capture(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)["X-Request-Id"] = request_id
                if context.experiment_run_id:
                    MutableHeaders(scope=message)["X-Experiment-Run-ID"] = (
                        context.experiment_run_id
                    )
            await send(message)

        if needs_lookup:
            override = await apply_experiment(state, context, degrade=experimental)

        try:
            if override is not None:
                await override(scope, receive, capture)
            else:
                await self.app(scope, receive, capture)
                if experimental and status_code < 400:
                    context.event = EXPERIMENTAL_ROUTES[(method, route)]
        finally:
            duration = perf_counter() - started
            metrics.requests.labels(method, route, str(status_code)).inc()
            metrics.duration.labels(route).observe(duration)
            if experimental or context.event == "stresser_lookup_failed":
                fields.update(
                    status_code=status_code,
                    latency_ms=round(duration * 1000, 3),
                    stress_level=context.stress_level,
                    degradation_type=context.degradation_type,
                )
                if context.error_type:
                    fields["error_type"] = context.error_type
                if context.delay_ms > 0:
                    state.logger.info(
                        "latency_injected", extra={"event_fields": fields}
                    )
                if status_code >= 500:
                    log_level = logging.ERROR
                elif status_code >= 400:
                    log_level = logging.WARNING
                else:
                    log_level = logging.INFO
                state.logger.log(
                    log_level, context.event, extra={"event_fields": fields}
                )
