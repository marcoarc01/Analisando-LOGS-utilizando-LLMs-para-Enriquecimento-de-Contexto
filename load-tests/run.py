"""Run one k6 execution with an atomic Docker name lock and no report files."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
LOCK = "observability-k6-run"
OWNER_LABEL = "observability.k6.owner"


def docker(*args, **kwargs):
    return subprocess.run(["docker", *args], cwd=ROOT, text=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testid", required=True)
    parser.add_argument(
        "--scenario",
        choices=[
            "baseline",
            "stress_cpu",
            "recovery",
            "network_latency",
            "network_timeout",
            "combined",
        ],
        required=True,
    )
    args = parser.parse_args()
    if not re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}",
        args.testid,
    ):
        parser.error("testid must be a neutral UUID v4")
    running = docker(
        "ps",
        "--format",
        "{{.Names}} {{.Image}} {{.Labels}}",
        capture_output=True,
        check=True,
    ).stdout
    if any(
        "grafana/k6" in line or "com.docker.compose.service=k6" in line
        for line in running.splitlines()
    ):
        parser.error("another k6 execution is active")
    owner = str(uuid.uuid4())
    # The fixed container name is acquired atomically by Docker; concurrent
    # invocations cannot both start. Direct Docker commands can bypass this guard.
    env = {
        **os.environ,
        "K6_EXPERIMENT_RUN_ID": args.testid,
        "K6_SCENARIO": args.scenario,
    }
    try:
        result = docker(
            "compose",
            "--profile",
            "loadtest",
            "run",
            "--rm",
            "--no-deps",
            "-T",
            "--name",
            LOCK,
            "--label",
            f"{OWNER_LABEL}={owner}",
            "k6",
            env=env,
        )
        return result.returncode
    finally:
        current = docker("inspect", LOCK, capture_output=True)
        if current.returncode == 0:
            container = json.loads(current.stdout)[0]
            if container["Config"]["Labels"].get(OWNER_LABEL) == owner:
                docker("stop", "--time", "6", LOCK, capture_output=True, check=True)
                # --rm removes only this ephemeral container; no volumes removed.
                remaining = docker("inspect", LOCK, capture_output=True)
                if remaining.returncode == 0:
                    docker("rm", LOCK, capture_output=True, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
