"""Minimal OpenAI-compatible chat client for Gemini, Groq and Ollama."""
import os
import threading
import time
from dataclasses import dataclass, field

import httpx

from app.config import PROVIDERS, Settings


class LLMError(RuntimeError):
    """Raised when no configured provider could produce a completion."""


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict]  # raw OpenAI-format tool calls
    provider: str
    model: str
    usage: dict = field(default_factory=dict)


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

        errors = []
        for provider in providers:
            try:
                return self._call(provider, messages, tools, response_format)
            except LLMError as exc:
                errors.append(f"{provider}: {exc}")
        raise LLMError("; ".join(errors))

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

            # Free tiers rate-limit aggressively; back off and retry.
            if resp.status_code in (429, 500, 502, 503) and attempt < retries - 1:
                retry_after = resp.headers.get("retry-after")
                time.sleep(min(float(retry_after), 60) if retry_after else min(2 ** (attempt + 1), 60))
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
