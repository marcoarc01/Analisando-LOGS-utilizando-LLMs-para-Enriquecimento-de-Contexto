import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    stresser_base_url: HttpUrl = "http://stresser-app:8080"
    stresser_timeout_seconds: float = Field(default=2, gt=0, allow_inf_nan=False)
    experiment_seed: int = 42
    max_experiment_delay_ms: int = Field(default=2000, ge=0)
    # "off" keeps the Go lookup but never delays or fails requests on purpose.
    degradation_policy: Literal["synthetic", "off"] = "synthetic"
    # Fixed CPU work per successful experimental request (0 = none).
    request_work_iterations: int = Field(default=0, ge=0)
    # Calibration only: append each Go lookup duration to this file inside the
    # container (never to stdout, so it never reaches Loki or a dataset).
    lookup_timing_file: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            **{
                name: os.environ[name.upper()]
                for name in cls.model_fields
                if name.upper() in os.environ
            }
        )
