import json
import logging
import runpy
from datetime import datetime, timedelta

import pytest

from app.telemetry import JsonFormatter, MESSAGES, create_logger, server_log_config


def test_logger_emits_json_to_stdout(capsys):
    logger = create_logger()
    logger.info("health_checked", extra={"event_fields": {"status_code": 200}})
    output = capsys.readouterr()
    assert output.err == ""
    record = json.loads(output.out)
    assert record["service"] == "application-api"
    assert record["event"] == "health_checked"
    assert record["level"] == "info"
    assert record["status_code"] == 200
    assert record["timestamp"].endswith("+00:00")


@pytest.mark.parametrize("event", MESSAGES)
def test_all_request_events_share_schema(event, capsys):
    create_logger().info(event, extra={"event_fields": {"request_id": "id"}})
    line = capsys.readouterr().out
    assert line.startswith("{") and len(line.splitlines()) == 1
    record = json.loads(line)
    assert record["event"] == event
    assert record["message"] and record["level"] == "info"
    assert datetime.fromisoformat(record["timestamp"]).utcoffset() == timedelta(0)
    assert not {"time", "msg", "status", "duration_ms"} & record.keys()
    assert not any(
        word in record["message"].lower()
        for word in ("normal", "anomaly", "anomalous", "stress80", "stress100")
    )


@pytest.mark.parametrize(
    "message,level,event",
    [
        ("Application startup complete.", logging.INFO, "service_started"),
        ("Application shutdown complete.", logging.INFO, "service_stopped"),
        (
            "token=private cookie=private Authorization=private body=private",
            logging.ERROR,
            "server_error",
        ),
    ],
)
def test_uvicorn_logs_sanitize_free_text(message, level, event):
    record = logging.LogRecord("uvicorn.error", level, "", 0, message, (), None)
    output = JsonFormatter().format(record)
    parsed = json.loads(output)
    assert parsed["event"] == event
    assert "private" not in output
    assert parsed["level"] == logging.getLevelName(level).lower()
    assert server_log_config()["handlers"]["stdout"]["stream"] == "ext://sys.stdout"


def test_server_entrypoint_uses_json_logs(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "uvicorn.run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    runpy.run_module("app.server", run_name="__main__")
    args, options = calls[0]
    assert args == ("app.main:app",)
    assert options["timeout_keep_alive"] == 30
    assert options["access_log"] is False
    assert options["log_config"]["formatters"]["json"]["()"] is JsonFormatter
