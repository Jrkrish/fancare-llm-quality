"""Minimal OpenAI-compatible chat client for Gemini, Groq and Ollama."""
import os
import re
import threading
import time
from dataclasses import dataclass, field

import httpx

from app.config import PROVIDERS, Settings

# A daily quota will not recover within a retry window, so there is no point waiting for it.
_DAILY_QUOTA_RE = re.compile(r"PerDay|per day|\(RPD\)|\(TPD\)", re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
MAX_WAIT_S = 60


class LLMError(RuntimeError):
    """Raised when no configured provider could produce a completion."""


class QuotaExhausted(LLMError):
    """The provider's daily quota is used up; retrying before it resets is pointless."""


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict]  # raw OpenAI-format tool calls
    provider: str
    model: str
    usage: dict = field(default_factory=dict)


def retry_delay(resp: httpx.Response, attempt: int) -> float:
    """Seconds to wait before retrying: Retry-After header, then Gemini's retryDelay, then exponential backoff."""
    header = resp.headers.get("retry-after")
    if header:
        try:
            return min(float(header), MAX_WAIT_S)
        except ValueError:
            pass
    match = _RETRY_DELAY_RE.search(resp.text)
    if match:
        return min(float(match.group(1)), MAX_WAIT_S)
    return min(2 ** (attempt + 1), MAX_WAIT_S)


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._last_call = 0.0

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None, response_format: dict | None = None
    ) -> LLMResponse:
        providers = [self.settings.llm_provider]
        if self.settings.llm_fallback_provider and self.settings.llm_fallback_provider not in providers:
            providers.append(self.settings.llm_fallback_provider)

        errors, quota_hit = [], False
        for provider in providers:
            try:
                return self._call(provider, messages, tools, response_format)
            except QuotaExhausted as exc:
                quota_hit = True
                errors.append(f"{provider}: {exc}")
            except LLMError as exc:
                errors.append(f"{provider}: {exc}")
        raise (QuotaExhausted if quota_hit and len(errors) == 1 else LLMError)("; ".join(errors))

    def _throttle(self) -> None:
        """Space requests at least llm_min_interval_s apart (free tiers limit requests per minute)."""
        with self._lock:
            wait = self._last_call + self.settings.llm_min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _call(
        self, provider: str, messages: list[dict], tools: list[dict] | None, response_format: dict | None
    ) -> LLMResponse:
        if provider not in PROVIDERS:
            raise LLMError(f"unknown provider '{provider}'")
        cfg = PROVIDERS[provider]
        headers = {}
        if cfg["key_env"]:
            key = os.getenv(cfg["key_env"])
            if not key:
                raise LLMError(f"{cfg['key_env']} is not set")
            headers["Authorization"] = f"Bearer {key}"

        model = self.settings.model_for(provider)
        body = {"model": model, "messages": messages, "temperature": self.settings.llm_temperature}
        if tools:
            body["tools"] = tools
        if response_format:
            body["response_format"] = response_format

        retries = self.settings.llm_max_retries
        for attempt in range(retries):
            self._throttle()
            try:
                resp = httpx.post(
                    f"{cfg['base_url']}/chat/completions",
                    json=body,
                    headers=headers,
                    timeout=self.settings.llm_timeout_s,
                )
            except httpx.HTTPError as exc:
                if attempt == retries - 1:
                    raise LLMError(f"request failed: {exc}") from exc
                time.sleep(2**attempt)
                continue

            if resp.status_code == 429 and _DAILY_QUOTA_RE.search(resp.text):
                raise QuotaExhausted(f"HTTP 429 daily quota exhausted for {model}: {resp.text[:300]}")
            # Rate limits and server-side errors (incl. Cloudflare 52x) are usually transient.
            if (resp.status_code == 429 or resp.status_code >= 500) and attempt < retries - 1:
                time.sleep(retry_delay(resp, attempt))
                continue
            if resp.status_code != 200:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            message = data["choices"][0]["message"]
            return LLMResponse(
                content=message.get("content") or "",
                tool_calls=message.get("tool_calls") or [],
                provider=provider,
                model=model,
                usage=data.get("usage") or {},
            )
        raise LLMError("retries exhausted")
