import json
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.stresser_client import StresserClient, StresserUnavailable
from app.telemetry import create_logger

RUN_ID = str(uuid.uuid4())
NEW_EVENTS = {"dependency_connection_opened", "dependency_connection_closed", "dependency_request_sent",
              "dependency_response_received", "dependency_response_failed"}
ALLOWED_KEYS = {"timestamp", "level", "service", "event", "message", "request_id", "experiment_run_id", "error"}


class StateHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        time.sleep(self.server.delay)
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
    httpd.delay, httpd.daemon_threads = 0.0, True
    httpd.handle_error = lambda *args: None
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def stresser_for(port, logs, timeout=1.0):
    stresser = StresserClient(Settings(stresser_base_url=f"http://127.0.0.1:{port}",
                                       stresser_timeout_seconds=timeout))
    stresser.logger = create_logger()
    stresser.logger.handlers[0].setStream(logs)
    return stresser


def records(logs):
    return [json.loads(line) for line in logs.getvalue().splitlines()]


def assert_no_leak(record, port):
    assert set(record) <= ALLOWED_KEYS
    values = {k: str(v) for k, v in record.items() if k not in ("timestamp", "request_id", "experiment_run_id")}
    text = " ".join(values.values()).lower()
    for forbidden in ("127.0.0.1", str(port), "localhost", "http", "api/state", "stress", "toxi"):
        assert forbidden not in text
    for key, value in values.items():
        assert not any(ch.isdigit() for ch in value), (key, value)


@pytest.mark.anyio
async def test_success_opens_once_and_reuses_the_connection(server):
    logs = StringIO()
    stresser = stresser_for(server.server_port, logs)
    assert await stresser.get_level(request_id="req-a", experiment_run_id=RUN_ID) == 0
    assert await stresser.get_level(request_id="req-b", experiment_run_id=RUN_ID) == 0
    await stresser.aclose()
    events = [(r["event"], r["request_id"]) for r in records(logs)]
    assert events == [("dependency_connection_opened", "req-a"), ("dependency_request_sent", "req-a"),
                      ("dependency_response_received", "req-a"),
                      ("dependency_request_sent", "req-b"), ("dependency_response_received", "req-b")]
    for record in records(logs):
        assert record["experiment_run_id"] == RUN_ID
        assert record["service"] == "application-api"
        assert record["level"] == "info"
        assert_no_leak(record, server.server_port)


@pytest.mark.anyio
async def test_timeout_fails_closes_and_reconnects(server):
    logs = StringIO()
    stresser = stresser_for(server.server_port, logs, timeout=0.05)
    server.delay = 0.3
    with pytest.raises(StresserUnavailable) as raised:
        await stresser.get_level(request_id="req-slow", experiment_run_id=RUN_ID)
    assert raised.value.error_type == "ReadTimeout"
    server.delay = 0.0
    assert await stresser.get_level(request_id="req-next", experiment_run_id=RUN_ID) == 0
    await stresser.aclose()
    events = [(r["event"], r["request_id"], r.get("error")) for r in records(logs)]
    assert events == [
        ("dependency_connection_opened", "req-slow", None), ("dependency_request_sent", "req-slow", None),
        ("dependency_response_failed", "req-slow", "ReadTimeout"), ("dependency_connection_closed", "req-slow", None),
        ("dependency_connection_opened", "req-next", None), ("dependency_request_sent", "req-next", None),
        ("dependency_response_received", "req-next", None),
    ]
    failed = next(r for r in records(logs) if r["event"] == "dependency_response_failed")
    assert failed["level"] == "warning" and failed["message"] == "Dependency response failed"
    for record in records(logs):
        assert_no_leak(record, server.server_port)


@pytest.mark.anyio
async def test_refused_connection_fails_without_close_event():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    logs = StringIO()
    stresser = stresser_for(port, logs)
    with pytest.raises(StresserUnavailable):
        await stresser.get_level(request_id="req-x")
    await stresser.aclose()
    [record] = records(logs)
    assert (record["event"], record["error"]) == ("dependency_response_failed", "ConnectError")
    assert "experiment_run_id" not in record
    assert_no_leak(record, port)


def test_events_follow_the_incoming_request(server):
    app = create_app(Settings(stresser_base_url=f"http://127.0.0.1:{server.server_port}", degradation_policy="off"))
    logs = StringIO()
    with TestClient(app) as client:
        app.state.logger.handlers[0].setStream(logs)
        response = client.get("/products", headers={"X-Request-ID": "req-api", "X-Experiment-Run-ID": RUN_ID})
    assert response.status_code == 200
    lines = records(logs)
    assert [r["event"] for r in lines] == ["dependency_connection_opened", "dependency_request_sent",
                                           "dependency_response_received", "products_completed"]
    assert {(r["request_id"], r["experiment_run_id"]) for r in lines} == {("req-api", RUN_ID)}
    for record in lines:
        if record["event"] in NEW_EVENTS:
            assert_no_leak(record, server.server_port)
