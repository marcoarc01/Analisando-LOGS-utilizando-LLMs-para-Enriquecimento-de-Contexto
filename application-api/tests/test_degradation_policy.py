import json

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.degradation import DegradationPolicy


def call(client, path):
    if path == "/orders":
        return client.post(path, json={"product_id": 1, "quantity": 2})
    return client.get(path)


def records(env):
    return [json.loads(line) for line in env.logs.getvalue().splitlines()]


def test_policy_setting_default_and_environment(monkeypatch):
    assert Settings().degradation_policy == "synthetic"
    monkeypatch.setenv("DEGRADATION_POLICY", "off")
    assert Settings.from_env().degradation_policy == "off"
    with pytest.raises(ValidationError):
        Settings(degradation_policy="random")


def test_synthetic_policy_is_the_default(make_environment):
    env = make_environment()
    assert isinstance(env.app.state.policy, DegradationPolicy)
    env.level = 100
    assert call(env.client, "/products").status_code == 500
    env.sleep.assert_awaited_once()
    assert [r["event"] for r in records(env)] == ["latency_injected", "experimental_500"]


@pytest.mark.parametrize(("path", "status", "event"), [
    ("/products", 200, "products_completed"),
    ("/orders", 201, "order_created"),
    ("/health", 200, "health_checked"),
])
@pytest.mark.parametrize("level", [0, 75, 92, 100])
def test_off_policy_never_degrades(make_environment, path, status, event, level):
    env = make_environment(degradation_policy="off")
    assert env.app.state.policy is None
    env.level = level
    response = call(env.client, path)
    assert response.status_code == status
    assert env.calls == 1
    env.sleep.assert_not_awaited()
    [record] = records(env)
    assert record["event"] == event
    assert record["stress_level"] == level
    assert record["degradation_type"] == "none"
    metrics = env.app.state.metrics
    assert metrics.errors_500._value.get() == metrics.errors_503._value.get() == 0


@pytest.mark.parametrize("error", [httpx.ReadTimeout("x"), httpx.ConnectError("x")])
def test_off_policy_keeps_real_lookup_failures(make_environment, error):
    env = make_environment(degradation_policy="off")
    env.error = error
    response = call(env.client, "/products")
    assert response.status_code == 503
    failed, record = records(env)
    assert (failed["event"], failed["error"]) == ("dependency_response_failed", type(error).__name__)
    assert record["event"] == "stresser_lookup_failed"
    assert record["error_type"] == type(error).__name__
    assert env.app.state.metrics.lookup_failures._value.get() == 1
