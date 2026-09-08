import json

from app.telemetry import create_logger


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
