"""DeepEval judge model backed by a free provider (Gemini, Groq or Ollama).

DeepEval defaults to OpenAI as its judge, which is paid. This wrapper reuses the
app's OpenAI-compatible client, asks for JSON output and parses it into the
pydantic schema DeepEval passes in.
"""
import asyncio
import json
import re
from dataclasses import replace

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.llm import LLMClient, QuotaExhausted


def _extract_json(text: str):
    # Reasoning models (e.g. Qwen3) may put their thinking before the JSON.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def _coerce(schema: type[BaseModel], data):
    """Some models return a bare list where DeepEval expects {"verdicts": [...]}; wrap it."""
    if isinstance(data, list):
        list_fields = [name for name, f in schema.model_fields.items() if "list" in str(f.annotation).lower()]
        if len(list_fields) == 1:
            data = {list_fields[0]: data}
    return schema.model_validate(data)


class JudgeLLM(DeepEvalBaseLLM):
    def __init__(self, provider: str, model: str, min_interval_s: float = 0.0, max_tokens: int = 2048):
        self.provider = provider
        self.model_id = model
        self.settings = replace(
            Settings(),
            llm_provider=provider,
            llm_model=model,
            llm_fallback_provider="",
            llm_temperature=0.0,
            llm_min_interval_s=min_interval_s,
            llm_max_retries=5,
            llm_max_tokens=max_tokens,
        )
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.exhausted: str | None = None  # set once the judge's daily quota runs out
        super().__init__(model)

    def load_model(self):
        return LLMClient(self.settings)

    def get_model_name(self) -> str:
        return f"{self.provider}/{self.model_id}"

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        # Circuit breaker: after a daily-quota error every further call would fail too, so fail fast.
        if self.exhausted:
            raise QuotaExhausted(self.exhausted)
        last_error = None
        for _ in range(2):  # one retry on unparseable output
            self.calls += 1
            try:
                resp = self.model.chat(
                    [{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"} if schema else None,
                )
            except QuotaExhausted as exc:
                self.exhausted = f"judge quota exhausted: {exc}"[:300]
                raise
            self.prompt_tokens += int(resp.usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(resp.usage.get("completion_tokens") or 0)
            if schema is None:
                return resp.content
            try:
                return _coerce(schema, _extract_json(resp.content))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
        raise ValueError(f"judge returned output that does not match {schema.__name__}: {last_error}")

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return await asyncio.to_thread(self.generate, prompt, schema)
