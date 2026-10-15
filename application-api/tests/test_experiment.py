import json
import math

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from app.degradation import DegradationPolicy


def experimental_call(client, path, headers=None):
    if path == "/orders":
        return client.post(path, json={"product_id": 1, "quantity": 2}, headers=headers)
    return client.get(path, headers=headers)


@pytest.mark.parametrize("path", ["/products", "/orders", "/health"])
@pytest.mark.parametrize("level", [0, 25, 50, 75, 100])
def test_experimental_routes(environment, path, level):
    environment.level = level
    response = experimental_call(environment.client, path)
    assert environment.calls == 1
    expected = 500 if level == 100 else 201 if path == "/orders" else 200
    assert response.status_code == expected
    if level <= 25:
        environment.sleep.assert_not_awaited()
    else:
        environment.sleep.assert_awaited_once_with(pytest.approx(2 * (level - 25) / 75))


@pytest.mark.parametrize(("seed", "status"), [(42, 500), (1, 503)])
@pytest.mark.parametrize("path", ["/products", "/orders", "/health"])
def test_controlled_errors(environment, seed, status, path):
    environment.level = 100
    environment.app.state.policy = DegradationPolicy(seed=seed)
    response = experimental_call(environment.client, path)
    assert response.status_code == status
    metrics = environment.app.state.metrics
    counter = metrics.errors_500 if status == 500 else metrics.errors_503
    assert counter._value.get() == 1
    assert metrics.lookup_failures._value.get() == 0


@pytest.mark.parametrize("path", ["/products", "/orders", "/health", "/ready"])
@pytest.mark.parametrize(
    "error", [httpx.ConnectError("secret-host"), httpx.ReadTimeout("secret-token")]
)
def test_unavailable_dependency(environment, path, error):
    environment.error = error
    response = experimental_call(environment.client, path)
    assert response.status_code == 503
    assert response.json() == {"detail": "Dependência indisponível"}
    environment.sleep.assert_not_awaited()
    assert environment.calls == 1
    assert environment.app.state.metrics.lookup_failures._value.get() == 1
    assert environment.app.state.metrics.errors_503._value.get() == 0
    assert math.isnan(environment.app.state.metrics.stress._value.get())
    output = environment.logs.getvalue()
    assert "secret-" not in output
    record = json.loads(output)
    assert record["event"] == "stresser_lookup_failed"
    assert record["level"] == "error"
    assert record["message"] == "Dependency lookup failed"
    assert record["error_type"] == type(error).__name__


@pytest.mark.parametrize(
    "error_class",
    [
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
    ],
)
def test_lookup_failure_log_has_no_dependency_address(environment, error_class):
    environment.error = error_class(
        "GET http://toxiproxy:8666/api/state via stresser-app:8080 failed"
    )
    environment.client.get("/products", headers={"X-Request-ID": "req-1"})
    record = json.loads(environment.logs.getvalue())
    assert record["error_type"] == error_class.__name__
    for key, value in record.items():
        if key == "timestamp":
            continue
        text = f"{key}={value}".lower()
        for forbidden in ("toxiproxy", "8666", "8080", "stresser-app", "http", "://"):
            assert forbidden not in text, (key, value)


@pytest.mark.parametrize(
    ("level", "path", "expected"),
    [(0, "/products", "info"), (100, "/products", "error")],
)
def test_log_level_by_status(environment, level, path, expected):
    environment.level = level
    experimental_call(environment.client, path)
    records = [json.loads(line) for line in environment.logs.getvalue().splitlines()]
    assert records[-1]["level"] == expected
    assert "error_type" not in records[-1]


@pytest.mark.parametrize(
    "path", ["/live", "/metrics", "/docs", "/redoc", "/openapi.json"]
)
def test_operational_routes_do_not_query_go_or_delay(environment, path):
    environment.level = 100
    environment.error = httpx.ConnectError("unavailable")
    assert environment.client.get(path).status_code == 200
    assert environment.calls == 0
    environment.sleep.assert_not_awaited()


def test_ready_at_full_stress(environment):
    environment.level = 100
    assert environment.client.get("/ready").json() == {"status": "ok"}
    assert environment.calls == 1
    environment.sleep.assert_not_awaited()
    assert environment.client.get("/products").status_code == 500


def test_logs_json_and_correlation(environment):
    environment.level = 50
    response = environment.client.get(
        "/products?secret=do-not-log",
        headers={
            "X-Experiment-Run-Id": "tcc-run-001",
            "Authorization": "private-token",
            "Cookie": "private-cookie",
        },
    )
    output = environment.logs.getvalue()
    assert all(
        value not in output
        for value in ["do-not-log", "private-token", "private-cookie"]
    )
    records = [json.loads(line) for line in output.splitlines()]
    assert {record["event"] for record in records} == {
        "latency_injected",
        "products_completed",
    }
    for record in records:
        assert set(record) == {
            "timestamp",
            "level",
            "service",
            "event",
            "message",
            "method",
            "route",
            "status_code",
            "latency_ms",
            "stress_level",
            "degradation_type",
            "request_id",
            "experiment_run_id",
        }
        assert record["experiment_run_id"] == "tcc-run-001"
        assert record["request_id"] == response.headers["X-Request-Id"]
        assert record["service"] == "application-api"
        assert record["stress_level"] == 50
        assert record["degradation_type"] == "delay"
        assert record["latency_ms"] >= 0


@pytest.mark.parametrize("run_id", ["", "x" * 129, "contains spaces"])
def test_invalid_run_id_omitted(environment, run_id):
    environment.client.get("/health", headers={"X-Experiment-Run-Id": run_id})
    assert "experiment_run_id" not in json.loads(environment.logs.getvalue())


@pytest.mark.parametrize(
    ("path", "event"), [("/orders", "order_created"), ("/health", "health_checked")]
)
def test_completion_events(environment, path, event):
    experimental_call(environment.client, path)
    assert json.loads(environment.logs.getvalue())["event"] == event


def test_metrics_have_bounded_labels(environment):
    for number in range(5):
        environment.client.get(
            f"/missing/{number}", headers={"X-Experiment-Run-Id": f"run-{number}"}
        )
        environment.client.request(f"CUSTOM{number}", "/missing")
    environment.client.get("/products", headers={"X-Experiment-Run-Id": "unique-id"})
    response = environment.client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    samples = [
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    ]
    for sample in samples:
        assert set(sample.labels) <= {
            "method",
            "route",
            "status",
            "degradation_type",
            "le",
        }
        assert all(
            "unique-id" not in value and "run-" not in value
            for value in sample.labels.values()
        )
    requests = [s for s in samples if s.name == "application_api_requests_total"]
    unknown = [s for s in requests if s.labels["route"] == "__unmatched__"]
    assert len(unknown) == 2
    assert {s.labels["method"] for s in unknown} == {"GET", "OTHER"}
    assert all(s.value == 5 for s in unknown)
    assert any(
        s.labels == {"method": "GET", "route": "/products", "status": "200"}
        and s.value == 1
        for s in requests
    )
