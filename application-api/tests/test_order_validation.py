import json
from unittest.mock import patch

import httpx
import pytest

from app.degradation import DegradationPolicy


def experimental_counters(metrics):
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for metric in (metrics.degradations, metrics.errors_500, metrics.errors_503)
        for family in metric.collect()
        for sample in family.samples
        if sample.name.endswith("_total")
    }


INVALID_ORDERS = [
    pytest.param("{", 422, id="malformed-json"),
    pytest.param("", 422, id="empty-body"),
    pytest.param(" ", 422, id="whitespace-body"),
    pytest.param("null", 422, id="null-body"),
    pytest.param("[]", 422, id="array-body"),
    pytest.param("{}", 422, id="missing-all-fields"),
    pytest.param('{"product_id":1}', 422, id="missing-quantity"),
    pytest.param('{"quantity":2}', 422, id="missing-product-id"),
    pytest.param('{"product_id":1,"quantity":2,"extra":1}', 422, id="unknown-field"),
    pytest.param('{"product_id":1,"quantity":"2"}', 422, id="string-quantity"),
    pytest.param('{"product_id":1,"quantity":2.0}', 422, id="float-quantity"),
    pytest.param('{"product_id":1,"quantity":true}', 422, id="boolean-quantity"),
    pytest.param('{"product_id":1,"quantity":null}', 422, id="null-quantity"),
    pytest.param('{"product_id":1,"quantity":0}', 422, id="zero-quantity"),
    pytest.param('{"product_id":1,"quantity":-1}', 422, id="negative-quantity"),
    pytest.param('{"product_id":"1","quantity":2}', 422, id="string-product-id"),
    pytest.param('{"product_id":true,"quantity":2}', 422, id="boolean-product-id"),
    pytest.param('{"product_id":1.0,"quantity":2}', 422, id="float-product-id"),
    pytest.param('{"product_id":1,"quantity":2} {}', 422, id="trailing-json"),
    pytest.param('{"product_id":999,"quantity":2}', 404, id="unknown-product"),
    pytest.param('{"product_id":0,"quantity":2}', 404, id="zero-product-id"),
    pytest.param('{"product_id":-1,"quantity":2}', 404, id="negative-product-id"),
]


@pytest.mark.parametrize("body,status", INVALID_ORDERS)
@pytest.mark.parametrize(
    "go_available", [True, False], ids=["stress-100", "go-offline"]
)
def test_invalid_orders_skip_experiment(environment, body, status, go_available):
    environment.level = 100
    if not go_available:
        environment.error = httpx.ConnectError("unavailable")
    policy = environment.app.state.policy
    metrics = environment.app.state.metrics
    random_before = policy.random.getstate()
    counters_before = experimental_counters(metrics)

    with patch.object(policy, "decide", wraps=policy.decide) as decide:
        response = environment.client.post(
            "/orders", content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status
        decide.assert_not_called()

    assert environment.calls == 0
    environment.sleep.assert_not_awaited()
    assert policy.random.getstate() == random_before
    assert experimental_counters(metrics) == counters_before
    assert metrics.lookup_failures._value.get() == 0
    assert (
        metrics.registry.get_sample_value(
            "application_api_requests_total",
            {"method": "POST", "route": "/orders", "status": str(status)},
        )
        == 1
    )
    record = json.loads(environment.logs.getvalue())
    if status == 404:
        assert record["event"] == "product_not_found"
        assert record["message"] == "Product not found"
    else:
        assert record["event"] == "request_rejected"
        assert record["message"] == "Request rejected"
    assert record["level"] == "warning"
    assert "error_type" not in record
    assert record["status_code"] == status
    assert record["stress_level"] is None
    assert record["degradation_type"] == "none"
    assert record["request_id"] == response.headers["X-Request-Id"]
    if status == 404:
        assert response.json() == {"detail": "Produto não encontrado"}


@pytest.mark.parametrize("seed", [42, 1])
def test_rejections_preserve_existing_counters_and_random_sequence(environment, seed):
    environment.level = 100
    policy = environment.app.state.policy = DegradationPolicy(seed=seed)
    reference = DegradationPolicy(seed=seed)
    metrics = environment.app.state.metrics

    for _ in range(3):
        valid = environment.client.post(
            "/orders", json={"product_id": 1, "quantity": 2}
        )
        assert valid.status_code == reference.decide(100).status_code
        calls_before = environment.calls
        random_before = policy.random.getstate()
        counters_before = experimental_counters(metrics)
        sleeps_before = environment.sleep.await_count

        assert environment.client.post("/orders", json={}).status_code == 422
        assert (
            environment.client.post(
                "/orders", json={"product_id": 999, "quantity": 2}
            ).status_code
            == 404
        )

        assert environment.calls == calls_before
        assert policy.random.getstate() == random_before
        assert experimental_counters(metrics) == counters_before
        assert environment.sleep.await_count == sleeps_before

    assert environment.calls == 3
