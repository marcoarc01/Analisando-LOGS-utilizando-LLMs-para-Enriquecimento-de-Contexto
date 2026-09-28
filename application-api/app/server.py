"""Entrypoint with structured Uvicorn lifecycle/error logs."""

import uvicorn

from app.telemetry import server_log_config

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        access_log=False,
        log_config=server_log_config(),
    )
