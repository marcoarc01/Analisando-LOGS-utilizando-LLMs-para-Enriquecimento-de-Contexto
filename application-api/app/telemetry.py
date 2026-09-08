import json
import logging
import sys
from datetime import datetime, timezone

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                **record.event_fields,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname.lower(),
                "service": "application-api",
                "event": record.getMessage(),
            },
            ensure_ascii=False,
        )


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
