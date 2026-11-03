import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from types import SimpleNamespace

import httpx
import pytest

from app.config import Settings
from app.stresser_client import DEADLINE_HEADER, SEND_HEADERS, StresserClient
from app.telemetry import create_logger


class StateHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.server.seen.append(self.headers.get(DEADLINE_HEADER))
        body = b'{"stress_level": 0, "cpu_workers": 0}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), StateHandler)
    httpd.seen, httpd.daemon_threads = [], True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def budget(value):
    assert value is not None and value.isdigit(), value
    return int(value)


@pytest.mark.anyio
async def test_header_sent_positive_and_within_the_timeout():
    seen = []

    async def handler(request):
        seen.append(request.headers.get(DEADLINE_HEADER))
        return httpx.Response(200, json={"stress_level": 0})

    for timeout, limit in ((0.292, 292), (2, 2000)):
        client = StresserClient(Settings(stresser_timeout_seconds=timeout), httpx.MockTransport(handler))
        try:
            await client.get_level(request_id="r")
        finally:
            await client.aclose()
        assert 1 <= budget(seen[-1]) <= limit


@pytest.mark.anyio
async def test_header_on_the_wire_and_never_logged(server):
    logs = StringIO()
    client = StresserClient(Settings(stresser_base_url=f"http://127.0.0.1:{server.server_port}",
                                     stresser_timeout_seconds=0.292))
    client.logger = create_logger()
    client.logger.handlers[0].setStream(logs)
    try:
        for i in range(3):
            assert await client.get_level(request_id=f"r{i}", experiment_run_id="run") == 0
    finally:
        await client.aclose()
    assert len(server.seen) == 3
    assert all(1 <= budget(v) <= 292 for v in server.seen)
    text = logs.getvalue().lower()
    assert "timeout-ms" not in text and "budget" not in text
    assert not any(f'"{v}"' in text or f": {v}" in text for v in server.seen)


def test_budget_subtracts_the_time_already_spent():
    client = StresserClient(Settings(stresser_timeout_seconds=0.292))
    assert client._budget_ms(time.perf_counter_ns()) in (291, 292)
    assert 180 <= client._budget_ms(time.perf_counter_ns() - 100_000_000) <= 192
    assert client._budget_ms(time.perf_counter_ns() - 10_000_000_000) == 1


@pytest.mark.anyio
async def test_value_recomputed_when_the_headers_are_written():
    client = StresserClient(Settings(stresser_timeout_seconds=0.292))
    request = SimpleNamespace(headers=[(b"Host", b"x"), (DEADLINE_HEADER.encode(), b"292"), (b"X-Request-ID", b"r")])
    state = {"failed": False, "started": time.perf_counter_ns() - 50_000_000}
    await client._tracer({"request_id": "r"}, state)(SEND_HEADERS, {"request": request})
    values = dict(request.headers)
    assert values[b"Host"] == b"x" and values[b"X-Request-ID"] == b"r"
    assert 230 <= int(values[DEADLINE_HEADER.encode()]) <= 242
    await client.aclose()
