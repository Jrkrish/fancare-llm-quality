"""Runtime settings.

Everything that the eval/regression suites need to vary (model, prompt version,
retriever) lives on Settings, so tests can build variants in-process:
    Settings(prompt_version="v2", retriever_mode="bm25")
Defaults come from environment variables (and .env if present).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# fastembed downloads models via huggingface_hub; Windows without symlink support warns on every load.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _env(name: str, default: str) -> str:
    """Environment value, treating an empty string (e.g. `X=` in .env) as unset."""
    return os.getenv(name) or default


# OpenAI-compatible chat endpoints for each supported (free) provider.
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
    },
    "ollama": {
        "base_url": _env("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1",
        "key_env": None,
        "default_model": "llama3.1:8b",
    },
}

# USD per 1M tokens (input, output). Reference list prices, used only to *estimate*
# what the traffic would cost on a paid tier; free-tier usage is billed $0. Values are
# the providers' published prices at model launch (2025): verify them on the provider
# pricing pages before quoting any cost figure. Unknown models are estimated at $0.
PRICING_PER_1M = {
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}

MAX_MESSAGE_CHARS = 2000


@dataclass
class Settings:
    # LLM
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "groq").lower())
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    llm_fallback_provider: str = field(default_factory=lambda: _env("LLM_FALLBACK_PROVIDER", "").lower())
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", ""))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.2")))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "30")))
    llm_max_retries: int = field(default_factory=lambda: int(_env("LLM_MAX_RETRIES", "3")))
    # Minimum gap between LLM requests; e.g. 2.0 keeps a client under 30 requests/minute.
    llm_min_interval_s: float = field(default_factory=lambda: float(_env("LLM_MIN_INTERVAL_S", "0")))
    # Cap on output tokens per request (0 = provider default). Groq reserves this against tokens-per-minute.
    llm_max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "0")))
    max_tool_rounds: int = 4

    # Prompt
    prompt_version: str = field(default_factory=lambda: _env("PROMPT_VERSION", "v1"))

    # Retrieval: mode is "hybrid" (BM25 + vector, RRF fused), "bm25" or "vector".
    retriever_mode: str = field(default_factory=lambda: _env("RETRIEVER_MODE", "hybrid").lower())
    rerank: bool = field(default_factory=lambda: _env("RERANK", "true").lower() == "true")
    retrieval_top_k: int = field(default_factory=lambda: int(_env("RETRIEVAL_TOP_K", "4")))
    # Cross-encoder logit below which a passage counts as "no evidence".
    rerank_min_score: float = field(default_factory=lambda: float(_env("RERANK_MIN_SCORE", "-4.0")))
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Tools: e.g. "get_order_status=timeout;check_warranty=malformed" or "*=empty"
    fault_injection: str = field(default_factory=lambda: _env("FAULT_INJECTION", ""))
    tool_timeout_s: float = field(default_factory=lambda: float(_env("TOOL_TIMEOUT_S", "3")))

    # Sessions
    max_history_turns: int = field(default_factory=lambda: int(_env("MAX_HISTORY_TURNS", "8")))

    # API
    app_api_key: str = field(default_factory=lambda: _env("APP_API_KEY", "dev-local-key"))
    rate_limit_per_min: int = field(default_factory=lambda: int(_env("RATE_LIMIT_PER_MIN", "30")))

    # Observability
    db_path: Path = field(default_factory=lambda: Path(_env("TRACE_DB_PATH", str(ROOT / "logs" / "traces.db"))))
    langfuse_public_key: str = field(default_factory=lambda: _env("LANGFUSE_PUBLIC_KEY", ""))
    langfuse_secret_key: str = field(default_factory=lambda: _env("LANGFUSE_SECRET_KEY", ""))
    langfuse_host: str = field(default_factory=lambda: _env("LANGFUSE_HOST", "https://cloud.langfuse.com"))

    # Paths
    docs_dir: Path = ROOT / "data" / "docs"
    data_dir: Path = ROOT / "data"
    prompts_dir: Path = ROOT / "prompts"
    model_cache_dir: Path = field(default_factory=lambda: Path(_env("MODEL_CACHE_DIR", str(ROOT / ".cache" / "models"))))

    def model_for(self, provider: str) -> str:
        if provider == self.llm_provider and self.llm_model:
            return self.llm_model
        if provider == "ollama" and self.ollama_model:
            return self.ollama_model
        return PROVIDERS[provider]["default_model"]

    @property
    def active_model(self) -> str:
        return self.model_for(self.llm_provider)


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = PRICING_PER_1M.get(model, (0.0, 0.0))
    return round((prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000, 6)
