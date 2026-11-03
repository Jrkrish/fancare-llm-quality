"""FanCare agent: guardrails -> LLM tool loop -> guardrails -> citations -> trace."""
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import date

from app.config import Settings, estimate_cost_usd
from app.guardrails import InputFilter, OutputFilter, Verdict
from app.llm import LLMClient, LLMError, LLMResponse
from app.memory import SessionMemory
from app.observability import LangfuseExporter, TraceRecord, TraceStore
from app.retrieval import HybridRetriever
from app.tools import TOOL_SCHEMAS, Toolbox, ToolRecord, ToolRunner

CITATION_RE = re.compile(r"\[(DOC-\d{2}:\d+)\]")
# Models cite in their own styles: [DOC-07:4], [DOC-07:4, DOC-12:2], or gpt-oss's
# full-width brackets 【DOC-07:4†L1-L3】.
_CITATION_GROUP_RE = re.compile(r"[\[【]([^\]】\n]*?DOC-\d{2}:\d+[^\]】\n]*?)[\]】]")
_DOC_ID_RE = re.compile(r"DOC-\d{2}:\d+")
BLOCKED_PLACEHOLDER = "[message blocked by guardrails]"
EMPTY_ANSWER_FALLBACK = "Sorry, I couldn't put together an answer just now. Please try again."


def normalize_citations(text: str) -> str:
    """Rewrite every citation style to the canonical [DOC-xx:n] form, one marker per chunk id."""
    return _CITATION_GROUP_RE.sub(lambda m: "".join(f"[{cid}]" for cid in _DOC_ID_RE.findall(m.group(1))), text)


class AgentUnavailable(RuntimeError):
    def __init__(self, trace_id: str, message: str):
        super().__init__(message)
        self.trace_id = trace_id


@dataclass
class Citation:
    chunk_id: str
    doc_id: str
    passage: str | None  # None if the chunk id does not exist in the corpus
    retrieved: bool  # True if search_docs returned this chunk during this turn


@dataclass
class AgentResult:
    answer: str
    citations: list[Citation]
    trace: TraceRecord
    retrieved: list[dict] = field(default_factory=list)  # passages the model saw this turn

    @property
    def tool_calls(self) -> list[ToolRecord]:
        return self.trace.tool_calls


def load_prompt(settings: Settings) -> str:
    return (settings.prompts_dir / settings.prompt_version / "system.txt").read_text(encoding="utf-8").strip()


class Agent:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient | None = None,
        retriever: HybridRetriever | None = None,
        trace_store: TraceStore | None = None,
    ):
        self.settings = settings
        self.retriever = retriever or HybridRetriever(settings)
        self.toolbox = Toolbox(settings, self.retriever)
        self.tools = ToolRunner(self.toolbox, settings)
        self.llm = llm or LLMClient(settings)
        self.memory = SessionMemory(settings.max_history_turns)
        self.system_prompt = load_prompt(settings)
        self.input_filter = InputFilter()
        self.output_filter = OutputFilter(self.system_prompt)
        self.store = trace_store or TraceStore(settings.db_path)
        self.exporter = LangfuseExporter(settings)

    def chat(self, session_id: str, message: str) -> AgentResult:
        start = time.perf_counter()
        checked = self.input_filter.check(message)
        trace = TraceRecord(
            trace_id=uuid.uuid4().hex,
            session_id=session_id,
            status="ok",
            prompt_version=self.settings.prompt_version,
            retriever_mode=self.settings.retriever_mode,
            user_message=checked.text,
            input_verdict=checked.verdicts[0].action,
            latency_ms=0,
            guardrail_events=list(checked.verdicts),
        )

        if checked.blocked:
            answer = self.input_filter.refusal(checked.category)
            trace.status, trace.answer = "blocked", answer
            self.memory.append(session_id, BLOCKED_PLACEHOLDER, answer)
            return self._finish(trace, start, AgentResult(answer, [], trace))

        system = f"{self.system_prompt}\n\nToday's date: {date.today().isoformat()}"
        notes = self.memory.notes(session_id)
        if notes:
            system += "\n\nFacts from documents the user uploaded in this session (validated fields only):\n- " + "\n- ".join(notes)
        messages = [
            {"role": "system", "content": system},
            *self.memory.history(session_id),
            {"role": "user", "content": checked.text},
        ]
        retrieved: dict[str, dict] = {}
        try:
            response = self._run_tool_loop(messages, trace, retrieved)
        except LLMError as exc:
            trace.status, trace.error = "error", str(exc)[:500]
            self._finish(trace, start, None)
            raise AgentUnavailable(trace.trace_id, "LLM provider unavailable") from exc

        raw_answer = normalize_citations(response.content.strip())
        if not raw_answer:
            raw_answer, trace.error = EMPTY_ANSWER_FALLBACK, "empty_completion"
        out = self.output_filter.check(raw_answer)
        trace.guardrail_events += out.verdicts
        trace.output_verdict = out.verdicts[0].action

        citations = [] if out.blocked else self._citations(out.text, retrieved)
        trace.answer = out.text
        trace.retrieved_chunk_ids = list(retrieved)
        trace.cited_chunk_ids = [c.chunk_id for c in citations]
        self.memory.append(session_id, checked.text, out.text)
        return self._finish(trace, start, AgentResult(out.text, citations, trace, list(retrieved.values())))

    def attach_document(self, session_id: str, result) -> None:
        """Make validated fields from an uploaded document available to later turns.

        Only the extracted, pattern-validated fields are stored; the document's
        free text never reaches the LLM, so instructions hidden in an upload are dropped.
        """
        if result.status == "failed":
            return
        known = ", ".join(f"{k}={v}" for k, v in result.fields.items() if v is not None)
        self.memory.add_note(session_id, f"{result.document_type}: {known}")

    # ------------------------------------------------------------------ helpers

    def _run_tool_loop(self, messages: list[dict], trace: TraceRecord, retrieved: dict[str, dict]) -> LLMResponse:
        for _ in range(self.settings.max_tool_rounds):
            response = self._llm_call(messages, trace, tools=TOOL_SCHEMAS)
            if not response.tool_calls:
                return response
            for call in response.tool_calls:
                call["id"] = call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            messages.append({"role": "assistant", "content": response.content or None, "tool_calls": response.tool_calls})
            for call in response.tool_calls:
                fn = call.get("function", {})
                record = self.tools.run(fn.get("name", ""), fn.get("arguments"))
                if record.name == "search_docs" and record.status == "ok":
                    self._screen_passages(record, trace)
                    retrieved.update({p["chunk_id"]: p for p in record.result["results"]})
                trace.tool_calls.append(record)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": record.llm_payload()})
        # Tool budget used up: force a final answer without tools.
        return self._llm_call(messages, trace, tools=None)

    def _llm_call(self, messages: list[dict], trace: TraceRecord, tools: list[dict] | None) -> LLMResponse:
        response = self.llm.chat(messages, tools=tools)
        trace.provider, trace.model = response.provider, response.model
        trace.prompt_tokens += int(response.usage.get("prompt_tokens") or 0)
        trace.completion_tokens += int(response.usage.get("completion_tokens") or 0)
        return response

    def _screen_passages(self, record: ToolRecord, trace: TraceRecord) -> None:
        """Drop retrieved passages that contain injection attempts (indirect prompt injection)."""
        kept = []
        for p in record.result["results"]:
            hit = self.input_filter.scan_passage(p["passage"])
            if hit:
                trace.guardrail_events.append(Verdict("retrieval", "block", "indirect_injection", f"{p['chunk_id']}: {hit[:80]}"))
            else:
                kept.append(p)
        record.result["results"] = kept

    def _citations(self, answer: str, retrieved: dict[str, dict]) -> list[Citation]:
        citations = []
        for chunk_id in dict.fromkeys(CITATION_RE.findall(answer)):
            chunk = self.retriever.by_id.get(chunk_id)
            citations.append(
                Citation(
                    chunk_id=chunk_id,
                    doc_id=chunk_id.split(":")[0],
                    passage=chunk.text if chunk else None,
                    retrieved=chunk_id in retrieved,
                )
            )
        return citations

    def _finish(self, trace: TraceRecord, start: float, result: AgentResult | None) -> AgentResult | None:
        trace.latency_ms = int((time.perf_counter() - start) * 1000)
        if trace.model:
            trace.cost_usd = estimate_cost_usd(trace.model, trace.prompt_tokens, trace.completion_tokens)
        self.store.write(trace)
        self.exporter.export(trace)
        return result
