import json
import logging
import sys
from datetime import datetime, timezone

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Control fields remain in raw logs for audit only. Future model inputs must
# exclude status_code, stress_level, degradation_type, experiment_run_id,
# request_id, scenario and testid (and event names that reveal the intervention).
MESSAGES = {
    "products_completed": "Product request completed",
    "order_created": "Order request completed",
    "health_checked": "Health request completed",
    "request_rejected": "Request rejected",
    "product_not_found": "Product not found",
    "latency_injected": "Request delayed",
    "experimental_500": "Request failed",
    "experimental_503": "Request unavailable",
    "stresser_lookup_failed": "Dependency lookup failed",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "event_fields", {})
        event = record.getMessage() if fields else "service_log"
        message = MESSAGES.get(event, "Service lifecycle updated")
        if record.name.startswith("uvicorn"):
            # Do not serialize arbitrary exception text, URLs, headers or bodies.
            event = (
                "server_error"
                if record.levelno >= logging.ERROR
                else "service_lifecycle"
            )
            message = (
                "Server operation failed"
                if record.levelno >= logging.ERROR
                else "Service lifecycle updated"
            )
            template = str(record.msg)
            if template == "Application startup complete.":
                event, message = "service_started", "Service started"
            elif template == "Application shutdown complete.":
                event, message = "service_stopped", "Service stopped"
        return json.dumps(
            {
                **fields,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname.lower(),
                "service": "application-api",
                "event": event,
                "message": message,
            },
            ensure_ascii=False,
        )


def server_log_config():
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"json": {"()": JsonFormatter}},
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "json",
            }
        },
        "loggers": {
            "uvicorn": {"handlers": ["stdout"], "level": "INFO", "propagate": False},
            "uvicorn.error": {
                "handlers": ["stdout"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {"handlers": [], "propagate": False},
        },
    }


def create_logger() -> logging.Logger:
    logger = logging.Logger("application-api", level=logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "application_api_requests_total",
            "HTTP requests completed",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "application_api_request_duration_seconds",
            "Full HTTP request duration including dependency and experiment delay",
            ["route"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10),
            registry=self.registry,
        )
        self.stress = Gauge(
            "application_api_observed_stress_level",
            "Last observed Go stress level; NaN when unknown or unavailable",
            registry=self.registry,
        )
        self.stress.set(float("nan"))
        self.lookup_failures = Counter(
            "application_api_stresser_failures_total",
            "Failed Go state lookups including readiness checks",
            registry=self.registry,
        )
        self.degradations = Counter(
            "application_api_degradations_total",
            "Applied experiment decisions excluding undegraded requests",
            ["degradation_type"],
            registry=self.registry,
        )
        self.errors_500 = Counter(
            "application_api_experimental_500_total",
            "Controlled experimental HTTP 500 responses",
            registry=self.registry,
        )
        self.errors_503 = Counter(
            "application_api_experimental_503_total",
            "Controlled experimental HTTP 503 responses excluding dependency failures",
            registry=self.registry,
        )
