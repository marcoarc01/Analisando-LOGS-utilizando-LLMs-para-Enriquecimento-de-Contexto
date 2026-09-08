import os

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    stresser_base_url: HttpUrl = "http://stresser-app:8080"
    stresser_timeout_seconds: float = Field(default=2, gt=0, allow_inf_nan=False)
    experiment_seed: int = 42
    max_experiment_delay_ms: int = Field(default=2000, ge=0)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            **{
                name: os.environ[name.upper()]
                for name in cls.model_fields
                if name.upper() in os.environ
            }
        )
