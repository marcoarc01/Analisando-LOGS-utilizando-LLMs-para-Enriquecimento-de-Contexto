import hashlib
import json

import pytest
from pydantic import ValidationError

import app.work
from app.config import Settings


@pytest.fixture
def work_calls(monkeypatch):
    calls = []
    original = app.work.chained_sha256

    def counting(iterations):
        calls.append(iterations)
        return original(iterations)

    monkeypatch.setattr(app.work, "chained_sha256", counting)
    return calls


def test_chained_sha256_is_deterministic():
    assert app.work.chained_sha256(0) == bytes(32)
    assert app.work.chained_sha256(1) == hashlib.sha256(bytes(32)).digest()
    assert app.work.chained_sha256(2) == hashlib.sha256(hashlib.sha256(bytes(32)).digest()).digest()


def test_work_setting_default_and_environment(monkeypatch):
    assert Settings().request_work_iterations == 0
    monkeypatch.setenv("REQUEST_WORK_ITERATIONS", "4000")
    assert Settings.from_env().request_work_iterations == 4000
    with pytest.raises(ValidationError):
        Settings(request_work_iterations=-1)


@pytest.mark.parametrize(("method", "path", "body", "status"), [
    ("GET", "/products", None, 200),
    ("GET", "/health", None, 200),
    ("POST", "/orders", {"product_id": 1, "quantity": 2}, 201),
])
def test_successful_experimental_routes_do_the_work(make_environment, work_calls, method, path, body, status):
    env = make_environment(degradation_policy="off", request_work_iterations=7)
    assert env.client.request(method, path, json=body).status_code == status
    assert work_calls == [7]


@pytest.mark.parametrize(("body", "status"), [({"product_id": 999, "quantity": 2}, 404),
                                              ({"product_id": 1, "quantity": 0}, 422)])
def test_client_errors_do_no_work(make_environment, work_calls, body, status):
    env = make_environment(degradation_policy="off", request_work_iterations=7)
    assert env.client.post("/orders", json=body).status_code == status
    assert work_calls == []


def test_default_does_no_work_and_logs_the_same_fields(make_environment, work_calls):
    default = make_environment()
    working = make_environment(request_work_iterations=50)
    for env in (default, working):
        assert env.client.get("/products").status_code == 200
    assert work_calls == [50]
    [plain] = [json.loads(line) for line in default.logs.getvalue().splitlines()]
    [worked] = [json.loads(line) for line in working.logs.getvalue().splitlines()]
    assert set(plain) == set(worked)


def test_synthetic_errors_skip_the_work(make_environment, work_calls):
    env = make_environment(request_work_iterations=7)
    env.level = 100
    assert env.client.get("/products").status_code == 500
    assert work_calls == []
