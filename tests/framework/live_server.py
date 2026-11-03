"""Starts the real app in a uvicorn subprocess for black-box tests."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveServer:
    def __init__(self, env: dict[str, str] | None = None, port: int | None = None):
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.env = {**os.environ, **(env or {})}
        self.proc: subprocess.Popen | None = None
        self.log_path = ROOT / "reports" / "server-logs" / f"server-{self.port}.log"

    def start(self, timeout: float = 120) -> "LiveServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=ROOT, env=self.env, stdout=self._log, stderr=subprocess.STDOUT,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early; see {self.log_path}")
            try:
                if requests.get(f"{self.url}/health", timeout=2).ok:
                    return self
            except requests.ConnectionError:
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"server did not become healthy in {timeout}s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "_log", None):
            self._log.close()


def server_env(**overrides: str) -> dict[str, str]:
    """Environment for a deterministic test server: no LLM key, isolated trace DB."""
    env = {
        "LLM_PROVIDER": "groq",
        "GROQ_API_KEY": "",
        "LLM_FALLBACK_PROVIDER": "",
        "APP_API_KEY": "dev-local-key",
        "RATE_LIMIT_PER_MIN": "1000",
        "TRACE_DB_PATH": str(ROOT / "logs" / "test_traces.db"),
    }
    env.update(overrides)
    return env
