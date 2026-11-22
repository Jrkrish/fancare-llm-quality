"""Record how much Groq quota is left, from the rate-limit headers of one tiny request.

    python -m tools.groq_quota

Groq reports its limits in response headers, and they cover different windows:
requests are per DAY, tokens are per MINUTE (there is no daily-token header). Right after a
busy run the token figure is therefore low and recovers within a minute; only the request
figure says anything about the day.

The probe sends one small chat request to the model the app uses, so it costs one request.
It reads the headers from error responses too (a 429 still carries them). It never fails the
job: on any problem it writes a note and exits 0. Output: reports/drift/groq_quota.md, plus a
`::warning::` line when under 20% of the day's requests are left.
"""
import os
import sys
from pathlib import Path

import httpx

from app.config import PROVIDERS, Settings
from eval.config import REPORTS_DIR

OUT = REPORTS_DIR / "drift" / "groq_quota.md"
LOW_FRACTION = 0.2


def _num(value: str | None) -> int | None:
    try:
        return int(float(value)) if value not in (None, "") else None
    except ValueError:
        return None


def parse_quota(headers) -> dict:
    """Pick the rate-limit fields out of Groq response headers (any mapping with .get)."""
    return {
        "limit_requests_per_day": _num(headers.get("x-ratelimit-limit-requests")),
        "remaining_requests_per_day": _num(headers.get("x-ratelimit-remaining-requests")),
        "reset_requests": headers.get("x-ratelimit-reset-requests"),
        "limit_tokens_per_minute": _num(headers.get("x-ratelimit-limit-tokens")),
        "remaining_tokens_per_minute": _num(headers.get("x-ratelimit-remaining-tokens")),
        "reset_tokens": headers.get("x-ratelimit-reset-tokens"),
        "retry_after": headers.get("retry-after"),
    }


def is_low(info: dict, fraction: float = LOW_FRACTION) -> bool:
    limit, left = info.get("limit_requests_per_day"), info.get("remaining_requests_per_day")
    return bool(limit) and left is not None and left / limit < fraction


def render(info: dict, model: str, status: int) -> str:
    def cell(key):
        value = info.get(key)
        return "n/a" if value is None else str(value)

    lines = [
        "## Groq quota",
        "",
        f"Probe: 1 request to `{model}` (HTTP {status}). The probe itself used one request.",
        "",
        "| | Left | Limit | Resets in |",
        "|---|---|---|---|",
        f"| Requests per **day** | {cell('remaining_requests_per_day')} | {cell('limit_requests_per_day')} | {cell('reset_requests')} |",
        f"| Tokens per **minute** | {cell('remaining_tokens_per_minute')} | {cell('limit_tokens_per_minute')} | {cell('reset_tokens')} |",
        "",
        "Groq exposes no daily token figure, and the per-minute token number is normally low just after the replay.",
    ]
    if info.get("retry_after"):
        lines.append(f"Rate limited: retry after {info['retry_after']}s.")
    if is_low(info):
        lines.append("**Low: under 20% of today's requests remain.**")
    return "\n".join(lines) + "\n"


def probe(settings: Settings | None = None) -> tuple[dict, str, int]:
    settings = settings or Settings()
    cfg = PROVIDERS["groq"]
    model = settings.model_for("groq")
    resp = httpx.post(
        f"{cfg['base_url']}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 16},
        headers={"Authorization": f"Bearer {os.environ[cfg['key_env']]}"},
        timeout=30,
    )
    return parse_quota(resp.headers), model, resp.status_code


def write(text: str, path: Path = OUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    if not os.getenv(PROVIDERS["groq"]["key_env"]):
        write("## Groq quota\n\nSkipped: GROQ_API_KEY is not set.\n")
        return 0
    try:
        info, model, status = probe()
    except Exception as exc:  # noqa: BLE001 - monitoring must never fail the job
        write(f"## Groq quota\n\nProbe failed: {type(exc).__name__}.\n")
        print(f"::warning title=Groq quota probe failed::{type(exc).__name__}")
        return 0
    write(render(info, model, status))
    if is_low(info):
        left, limit = info["remaining_requests_per_day"], info["limit_requests_per_day"]
        print(f"::warning title=Groq daily quota low::{left} of {limit} requests left today")
    print(f"Groq requests/day left: {info['remaining_requests_per_day']} of {info['limit_requests_per_day']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
