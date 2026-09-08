import httpx
import pytest

from app.config import Settings
from app.stresser_client import StresserClient, StresserUnavailable


@pytest.mark.anyio
async def test_success_reuses_client_and_timeout():
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.path == "/api/state"
        assert set(request.extensions["timeout"].values()) == {0.25}
        return httpx.Response(200, json={"stress_level": 25, "cpu_workers": 2})

    client = StresserClient(
        Settings(stresser_timeout_seconds=0.25), httpx.MockTransport(handler)
    )
    original = client.http
    try:
        assert await client.get_level() == 25
        assert await client.get_level() == 25
        assert client.http is original
        assert len(requests) == 2
    finally:
        await client.aclose()
    assert original.is_closed


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error_type",
    [
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
        httpx.RemoteProtocolError,
    ],
)
async def test_transport_failure_without_retry(error_type):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise error_type("internal host with private details", request=request)

    client = StresserClient(Settings(), httpx.MockTransport(handler))
    try:
        with pytest.raises(StresserUnavailable, match="^Dependência indisponível$"):
            await client.get_level()
        assert calls == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        b"{",
        b"null",
        b"[]",
        b"{}",
        b'{"stress_level":null}',
        b'{"stress_level":true}',
        b'{"stress_level":"50"}',
        b'{"stress_level":50.0}',
        b'{"stress_level":-1}',
        b'{"stress_level":101}',
    ],
)
async def test_invalid_response(body):
    client = StresserClient(
        Settings(),
        httpx.MockTransport(lambda request: httpx.Response(200, content=body)),
    )
    try:
        with pytest.raises(StresserUnavailable):
            await client.get_level()
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("status", [301, 404, 500, 503])
async def test_non_success_response(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={"Location": "http://other.invalid"})

    client = StresserClient(Settings(), httpx.MockTransport(handler))
    try:
        with pytest.raises(StresserUnavailable):
            await client.get_level()
        assert len(calls) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_default_transport_closes_without_connecting():
    client = StresserClient(Settings())
    await client.aclose()
    assert client.http.is_closed
