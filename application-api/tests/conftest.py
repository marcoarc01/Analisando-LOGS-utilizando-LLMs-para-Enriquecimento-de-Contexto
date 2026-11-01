from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.stresser_client import StresserClient


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def environment():
    env = SimpleNamespace(
        level=0, calls=0, requests=[], error=None, sleep=AsyncMock(), logs=StringIO()
    )

    def handler(request):
        env.calls += 1
        env.requests.append(request)
        if env.error:
            raise env.error
        return httpx.Response(200, json={"stress_level": env.level, "cpu_workers": 0})

    def factory(settings):
        env.stresser = StresserClient(settings, httpx.MockTransport(handler))
        return env.stresser

    env.app = create_app(Settings(), stresser_factory=factory, sleep=env.sleep)
    with TestClient(env.app) as client:
        env.app.state.logger.handlers[0].setStream(env.logs)
        env.client = client
        yield env
    assert env.stresser.http.is_closed


@pytest.fixture
def client(environment):
    return environment.client


@pytest.fixture
def make_environment():
    clients = []

    def build(**settings):
        env = SimpleNamespace(level=0, calls=0, error=None, sleep=AsyncMock(), logs=StringIO())

        def handler(request):
            env.calls += 1
            if env.error:
                raise env.error
            return httpx.Response(200, json={"stress_level": env.level, "cpu_workers": 0})

        app = create_app(
            Settings(**settings),
            stresser_factory=lambda config: StresserClient(config, httpx.MockTransport(handler)),
            sleep=env.sleep,
        )
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        app.state.logger.handlers[0].setStream(env.logs)
        env.app, env.client = app, client
        return env

    yield build
    for client in clients:
        client.__exit__(None, None, None)
