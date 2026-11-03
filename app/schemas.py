from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import MAX_MESSAGE_CHARS


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$", examples=["demo-session-1"])
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS, examples=["My fan shows E4. What should I do?"])

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str
    passage: str | None
    retrieved: bool


class ToolCallOut(BaseModel):
    name: str
    arguments: dict
    status: str
    error: str | None


class ChatResponse(BaseModel):
    session_id: str
    trace_id: str
    answer: str
    citations: list[CitationOut]
    tool_calls: list[ToolCallOut]


class ErrorResponse(BaseModel):
    # str for application errors; FastAPI's own 422 validation errors carry a list.
    detail: str | list[dict]
    trace_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    prompt_version: str
    provider: str
    model: str
    retriever_mode: str
    rerank: bool


class Turn(BaseModel):
    user: str
    assistant: str


class HistoryResponse(BaseModel):
    session_id: str
    turns: list[Turn]


class ExtractedFields(BaseModel):
    serial_no: str | None
    purchase_date: str | None
    model: str | None


class UploadResponse(BaseModel):
    status: Literal["ok", "partial", "failed"]
    document_type: Literal["warranty_card", "invoice", "unknown"]
    fields: ExtractedFields
    missing: list[str]
    warnings: list[str]
