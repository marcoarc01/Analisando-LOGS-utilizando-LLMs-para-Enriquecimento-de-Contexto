import asyncio
import json
from uuid import UUID

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from app.config import Settings
from app.stresser_client import StresserClient


@pytest.mark.parametrize("path", ["/products", "/orders", "/health", "/ready"])
@pytest.mark.parametrize("level", [0, 100])
def test_headers_propagate_to_go_response_and_logs(environment, path, level):
    environment.level = level
    headers = {
        "x-request-id": "request-123",
        "x-experiment-run-id": "test-baseline-001",
        "Authorization": "private-token",
        "Cookie": "private-cookie",
    }
    if path == "/orders":
        response = environment.client.post(
            path, json={"product_id": 1, "quantity": 2}, headers=headers
        )
    else:
        response = environment.client.get(path, headers=headers)
    assert response.headers["X-Request-ID"] == "request-123"
    assert response.headers["X-Experiment-Run-ID"] == "test-baseline-001"
    assert environment.calls == 1
    upstream = environment.requests[0]
    assert upstream.url.path == "/api/state"
    assert upstream.headers["X-Request-ID"] == "request-123"
    assert upstream.headers["X-Experiment-Run-ID"] == "test-baseline-001"
    assert "authorization" not in upstream.headers
    assert "cookie" not in upstream.headers
    records = [json.loads(line) for line in environment.logs.getvalue().splitlines()]
    if path != "/ready":
        assert records
    for record in records:
        assert record["request_id"] == "request-123"
        assert record["experiment_run_id"] == "test-baseline-001"


def test_dependency_failure_keeps_correlation(environment):
    environment.error = httpx.ConnectError("internal details")
    response = environment.client.get(
        "/health",
        headers={"X-Request-ID": "failure-001", "X-Experiment-Run-ID": "run-001"},
    )
    assert response.status_code == 503
    record = json.loads(environment.logs.getvalue())
    assert record["event"] == "stresser_lookup_failed"
    assert record["request_id"] == "failure-001"
    assert record["experiment_run_id"] == "run-001"
    assert environment.requests[0].headers["X-Request-ID"] == "failure-001"


@pytest.mark.parametrize("request_id", ["", "bad id", "x" * 129, "bad\tid"])
def test_invalid_request_id_replaced_and_forwarded(environment, request_id):
    response = environment.client.get("/health", headers={"X-Request-ID": request_id})
    generated = response.headers["X-Request-ID"]
    assert str(UUID(generated)) == generated
    assert generated != request_id
    assert environment.requests[0].headers["X-Request-ID"] == generated
    assert json.loads(environment.logs.getvalue())["request_id"] == generated


def test_missing_headers_generate_unique_ids_without_run_id_leakage(environment):
    environment.client.get(
        "/health", headers={"X-Request-ID": "supplied", "X-Experiment-Run-ID": "run-1"}
    )
    responses = [environment.client.get("/health") for _ in range(3)]
    ids = [response.headers["X-Request-ID"] for response in responses]
    assert len(set(ids)) == 3
    for request, response, request_id in zip(environment.requests[1:], responses, ids):
        assert str(UUID(request_id)) == request_id
        assert request.headers["X-Request-ID"] == request_id
        assert "X-Experiment-Run-ID" not in request.headers
        assert "X-Experiment-Run-ID" not in response.headers


@pytest.mark.parametrize("run_id", ["", "bad id", "x" * 129])
def test_invalid_run_id_never_forwarded(environment, run_id):
    response = environment.client.get(
        "/health", headers={"X-Experiment-Run-ID": run_id}
    )
    assert "X-Experiment-Run-ID" not in response.headers
    assert "X-Experiment-Run-ID" not in environment.requests[0].headers
    assert "experiment_run_id" not in json.loads(environment.logs.getvalue())


@pytest.mark.parametrize(
    "body,status", [("{}", 422), ('{"product_id":999,"quantity":2}', 404)]
)
def test_invalid_orders_keep_ids_without_querying_go(environment, body, status):
    response = environment.client.post(
        "/orders",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": "rejected-001",
            "X-Experiment-Run-ID": "validation-001",
        },
    )
    assert response.status_code == status
    assert environment.calls == 0
    record = json.loads(environment.logs.getvalue())
    assert record["request_id"] == response.headers["X-Request-ID"] == "rejected-001"
    assert record["experiment_run_id"] == "validation-001"


def test_ids_do_not_become_metric_labels(environment):
    environment.client.get(
        "/products",
        headers={"X-Request-ID": "unique-request", "X-Experiment-Run-ID": "unique-run"},
    )
    output = environment.client.get("/metrics").text
    assert "unique-request" not in output
    assert "unique-run" not in output
    for family in text_string_to_metric_families(output):
        for sample in family.samples:
            assert "request_id" not in sample.labels
            assert "experiment_run_id" not in sample.labels


@pytest.mark.anyio
async def test_shared_http_client_isolates_concurrent_request_headers():
    seen = []

    async def handler(request):
        await asyncio.sleep(0)
        seen.append(dict(request.headers))
        return httpx.Response(200, json={"stress_level": 0})

    client = StresserClient(Settings(), httpx.MockTransport(handler))
    try:
        levels = await asyncio.gather(
            *(
                client.get_level(
                    request_id=f"request-{i}", experiment_run_id=f"run-{i}"
                )
                for i in range(10)
            )
        )
        assert levels == [0] * 10
        assert {(h["x-request-id"], h["x-experiment-run-id"]) for h in seen} == {
            (f"request-{i}", f"run-{i}") for i in range(10)
        }
        await client.get_level()
        assert "x-request-id" not in seen[-1]
        assert "x-experiment-run-id" not in seen[-1]
    finally:
        await client.aclose()
