"""Root conftest: shared options, the LLM skip/rerun policy, quarantine, and flakiness recording.

Flakiness strategy
- Tests marked `llm` talk to a non-deterministic model. They are retried
  (pytest-rerunfailures) FLAKY_RERUNS times before counting as failed, and
  skipped when no API key is configured.
- Tests listed in tests/quarantine.txt still run but cannot fail the build
  (non-strict xfail); they stay visible in the report until fixed.
- Every outcome, including reruns, is appended to reports/flaky/history.jsonl;
  `python -m tools.flaky_report` ranks the flakiest tests across runs.
"""
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import PROVIDERS, Settings

ROOT = Path(__file__).resolve().parent.parent
QUARANTINE_FILE = Path(__file__).parent / "quarantine.txt"
FLAKY_HISTORY = ROOT / "reports" / "flaky" / "history.jsonl"
LLM_SUITE_MARKERS = {"llm", "eval", "safety", "tooluse", "conversation"}


def pytest_addoption(parser):
    group = parser.getgroup("fancare eval")
    group.addoption("--eval-profile", default="smoke", help="eval profile from eval/config.toml (smoke | nojudge | full)")
    group.addoption("--prompt-version", default=None, help="prompt version under test, e.g. v1")
    group.addoption("--eval-runs", type=int, default=None, help="runs per golden case (overrides the profile)")
    group.addoption("--eval-category", action="append", default=None, help="limit to a golden-case category (repeatable)")
    group.addoption("--suite-runs", type=int, default=None, help="attempts per case for safety/tool-use/conversation suites")


def llm_unavailable_reason() -> str | None:
    provider = Settings().llm_provider
    key_env = PROVIDERS.get(provider, {}).get("key_env")
    if key_env and not os.getenv(key_env):
        return f"{key_env} is not set (LLM_PROVIDER={provider})"
    return None


def _quarantined() -> list[str]:
    if not QUARANTINE_FILE.exists():
        return []
    lines = QUARANTINE_FILE.read_text(encoding="utf-8").splitlines()
    return [ln.split("#")[0].strip() for ln in lines if ln.split("#")[0].strip()]


def pytest_collection_modifyitems(config, items):
    reruns = int(os.getenv("FLAKY_RERUNS") or "1")
    no_llm = llm_unavailable_reason()
    quarantined = _quarantined()
    for item in items:
        if LLM_SUITE_MARKERS & set(item.keywords):
            if no_llm:
                item.add_marker(pytest.mark.skip(reason=f"needs an LLM: {no_llm}"))
            elif reruns and not item.get_closest_marker("flaky"):
                item.add_marker(pytest.mark.flaky(reruns=reruns, reruns_delay=5))
        if any(item.nodeid.startswith(q) for q in quarantined):
            item.add_marker(pytest.mark.xfail(reason="quarantined (tests/quarantine.txt)", strict=False))


# ------------------------------------------------------------ flakiness log

_outcomes: list[dict] = []


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _outcomes.append({"nodeid": report.nodeid, "outcome": report.outcome, "duration": round(report.duration, 3),
                          "quarantined": "quarantined" in str(getattr(report, "wasxfail", ""))})


def pytest_sessionfinish(session, exitstatus):
    if not _outcomes or os.getenv("FLAKY_RECORD", "1") == "0":
        return
    FLAKY_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    run = {"session": uuid.uuid4().hex[:8], "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
           "git_sha": os.getenv("GITHUB_SHA", "")}
    with FLAKY_HISTORY.open("a", encoding="utf-8") as f:
        for o in _outcomes:
            f.write(json.dumps({**run, **o}) + "\n")


# ------------------------------------------------------------ shared fixtures


@pytest.fixture(scope="session")
def harness(request):
    """In-process agent harness for the LLM suites (prompt version from --prompt-version or .env)."""
    from dataclasses import replace

    from eval.harness import AgentHarness

    settings = Settings()
    if request.config.option.prompt_version:
        settings = replace(settings, prompt_version=request.config.option.prompt_version)
    return AgentHarness(settings)


@pytest.fixture(scope="session")
def suite_runs(request) -> int:
    return request.config.option.suite_runs or int(os.getenv("SUITE_RUNS") or "1")
