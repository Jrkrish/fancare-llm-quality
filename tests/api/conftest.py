"""API suite fixtures.

By default a dedicated server is started with no LLM key, so every status code
and schema check is deterministic and free. Set FANCARE_BASE_URL to test an
already running deployment instead (rate-limit tests always use their own server).
"""
import os

import pytest

from tests.framework.api_client import ApiClient
from tests.framework.live_server import LiveServer, server_env
from tests.framework.openapi import OpenApiValidator


@pytest.fixture(scope="session")
def base_url():
    external = os.getenv("FANCARE_BASE_URL")
    if external:
        yield external
        return
    server = LiveServer(server_env()).start()
    yield server.url
    server.stop()


@pytest.fixture(scope="session")
def api(base_url) -> ApiClient:
    return ApiClient(base_url)


@pytest.fixture(scope="session")
def openapi(api) -> OpenApiValidator:
    return OpenApiValidator(api.openapi())


@pytest.fixture(scope="module")
def limited_api():
    server = LiveServer(server_env(RATE_LIMIT_PER_MIN="3")).start()
    yield ApiClient(server.url)
    server.stop()


@pytest.fixture(scope="session")
def llm_api():
    """A server with the real LLM configuration from .env (for `llm`-marked tests)."""
    from tests.framework.live_server import ROOT

    server = LiveServer({"TRACE_DB_PATH": str(ROOT / "logs" / "test_traces.db")}).start()
    yield ApiClient(server.url)
    server.stop()
