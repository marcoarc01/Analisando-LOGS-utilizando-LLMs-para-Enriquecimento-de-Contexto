import pytest
from pydantic import ValidationError

from app.config import Settings


def test_environment_settings(monkeypatch):
    monkeypatch.setenv("STRESSER_BASE_URL", "http://go.local:8080")
    monkeypatch.setenv("STRESSER_TIMEOUT_SECONDS", "0.5")
    monkeypatch.setenv("EXPERIMENT_SEED", "11")
    monkeypatch.setenv("MAX_EXPERIMENT_DELAY_MS", "100")
    settings = Settings.from_env()
    assert str(settings.stresser_base_url) == "http://go.local:8080/"
    assert settings.stresser_timeout_seconds == 0.5
    assert settings.experiment_seed == 11
    assert settings.max_experiment_delay_ms == 100


@pytest.mark.parametrize(
    "values",
    [
        {"stresser_base_url": "not-a-url"},
        {"stresser_timeout_seconds": 0},
        {"stresser_timeout_seconds": float("inf")},
        {"max_experiment_delay_ms": -1},
    ],
)
def test_invalid_settings(values):
    with pytest.raises(ValidationError):
        Settings(**values)
