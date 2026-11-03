"""Agent tools, their JSON schemas, and a runner with fault injection.

Every tool result is validated against a pydantic output model, so an injected
"malformed" or "empty" fault goes through the same code path a real broken
backend would.
"""
import itertools
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import Settings
from app.retrieval import HybridRetriever

ORDER_ID_RE = re.compile(r"^ORD-\d{5}$")
SERIAL_RE = re.compile(r"^[A-Z]{2}\d{2}-\d{4}-\d{5}$")
KNOWN_NEEDS = {"smart", "remote", "dc_motor", "light", "quiet", "outdoor", "flush_mount"}
FAULT_MODES = {"timeout", "malformed", "empty"}
PARTS = ("motor", "electronics", "accessories")

# ---------------------------------------------------------------- input models


class SearchDocsIn(BaseModel):
    query: str = Field(min_length=2, max_length=300)


class OrderStatusIn(BaseModel):
    order_id: str

    @field_validator("order_id")
    @classmethod
    def _format(cls, v: str) -> str:
        v = v.strip().upper()
        if not ORDER_ID_RE.match(v):
            raise ValueError("order_id must look like ORD-12345")
        return v


class CheckWarrantyIn(BaseModel):
    serial_no: str

    @field_validator("serial_no")
    @classmethod
    def _format(cls, v: str) -> str:
        v = v.strip().upper()
        if not SERIAL_RE.match(v):
            raise ValueError("serial_no must look like SW52-2403-00117")
        return v


class RecommendProductIn(BaseModel):
    room_size: float = Field(gt=0, le=5000, description="Room floor area in square feet")
    budget: float = Field(gt=0, le=10000, description="Maximum price in US dollars")
    needs: list[str] = Field(default_factory=list)


class CreateTicketIn(BaseModel):
    summary: str = Field(min_length=10, max_length=500)
    severity: Literal["low", "medium", "high", "critical"]


# --------------------------------------------------------------- output models


class Passage(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    status: str
    passage: str
    score: float | None


class SearchDocsOut(BaseModel):
    results: list[Passage]


class OrderStatusOut(BaseModel):
    order_id: str
    found: bool
    status: Literal["processing", "shipped", "out_for_delivery", "delivered", "cancelled"] | None = None
    product_id: str | None = None
    placed_on: date | None = None
    carrier: str | None = None
    tracking_id: str | None = None
    estimated_delivery: date | None = None
    delivered_on: date | None = None


class Coverage(BaseModel):
    until: date | None
    active: bool


class CheckWarrantyOut(BaseModel):
    serial_no: str
    found: bool
    model: str | None = None
    purchase_date: date | None = None
    policy_doc_id: str | None = None
    registered_within_60_days: bool | None = None
    commercial_use: bool | None = None
    void_reason: str | None = None
    coverage: dict[Literal["motor", "electronics", "accessories"], Coverage] | None = None


class ProductSummary(BaseModel):
    product_id: str
    name: str
    blade_span_in: int
    price_usd: float
    features: list[str]


class RecommendProductOut(BaseModel):
    products: list[ProductSummary]
    unrecognised_needs: list[str]


class CreateTicketOut(BaseModel):
    ticket_id: str
    severity: str
    status: Literal["open"]
    created_at: datetime


# ---------------------------------------------------------- schemas for the LLM


def _fn(name: str, description: str, model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}


TOOL_SCHEMAS = [
    _fn("search_docs", "Search the official Corvane documentation. Returns passages with chunk ids to cite.", SearchDocsIn),
    _fn("get_order_status", "Look up the status of an order by its ID (format ORD-12345).", OrderStatusIn),
    _fn("check_warranty", "Check warranty coverage for a fan by serial number (format SW52-2403-00117).", CheckWarrantyIn),
    _fn(
        "recommend_product",
        "Recommend fans from the catalogue. needs may include: " + ", ".join(sorted(KNOWN_NEEDS)) + ".",
        RecommendProductIn,
    ),
    _fn("create_support_ticket", "Open a support ticket. Do not include personal data in the summary.", CreateTicketIn),
]

# ------------------------------------------------------------- implementations


def _add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 Feb
        return d.replace(year=d.year + years, day=28)


def warranty_coverage(entry: dict, today: date) -> dict:
    """Apply DOC-12 (2025 policy) or DOC-13 (2022 policy) to a registry entry."""
    purchased = date.fromisoformat(entry["purchase_date"])
    if purchased >= date(2025, 1, 1):
        policy = "DOC-12"
        if entry["commercial_use"]:
            years = dict.fromkeys(PARTS, 1)
        else:
            years = {"motor": 5, "electronics": 3 if entry["registered_within_60_days"] else 2, "accessories": 1}
    elif purchased >= date(2022, 3, 1):
        policy = "DOC-13"
        # The 2022 policy did not cover commercial use at all.
        years = dict.fromkeys(PARTS, 0) if entry["commercial_use"] else {"motor": 3, "electronics": 1, "accessories": 1}
    else:
        return {"policy_doc_id": None, "coverage": {k: {"until": None, "active": False} for k in PARTS}}

    coverage = {}
    for part, n in years.items():
        until = _add_years(purchased, n)
        coverage[part] = {"until": until.isoformat(), "active": entry["void_reason"] is None and today < until}
    return {"policy_doc_id": policy, "coverage": coverage}


class Toolbox:
    def __init__(self, settings: Settings, retriever: HybridRetriever):
        self.retriever = retriever
        self.orders = {o["order_id"]: o for o in self._load(settings, "orders.json")}
        self.registry = {w["serial_no"]: w for w in self._load(settings, "warranty_registry.json")}
        self.catalogue = self._load(settings, "catalogue.json")
        self.tickets: list[dict] = []
        self._ticket_seq = itertools.count(20001)
        self._lock = threading.Lock()

    @staticmethod
    def _load(settings: Settings, name: str) -> list[dict]:
        return json.loads((settings.data_dir / name).read_text(encoding="utf-8"))

    def search_docs(self, query: str) -> dict:
        return {
            "results": [
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title, "section": c.section,
                 "status": c.status, "passage": c.text, "score": c.score}
                for c in self.retriever.search(query)
            ]
        }

    def get_order_status(self, order_id: str) -> dict:
        order = self.orders.get(order_id)
        return {**order, "found": True} if order else {"order_id": order_id, "found": False}

    def check_warranty(self, serial_no: str) -> dict:
        entry = self.registry.get(serial_no)
        if not entry:
            return {"serial_no": serial_no, "found": False}
        return {**entry, "found": True, **warranty_coverage(entry, date.today())}

    def recommend_product(self, room_size: float, budget: float, needs: list[str]) -> dict:
        wanted = {n.strip().lower() for n in needs}
        known, unknown = wanted & KNOWN_NEEDS, sorted(wanted - KNOWN_NEEDS)
        matches = [
            p for p in self.catalogue
            if p["room_sqft_min"] <= room_size <= p["room_sqft_max"]
            and p["price_usd"] <= budget
            and known <= set(p["features"])
        ]
        matches.sort(key=lambda p: p["price_usd"])
        return {
            "products": [
                {k: p[k] for k in ("product_id", "name", "blade_span_in", "price_usd", "features")} for p in matches[:3]
            ],
            "unrecognised_needs": unknown,
        }

    def create_support_ticket(self, summary: str, severity: str) -> dict:
        with self._lock:
            ticket = {
                "ticket_id": f"TKT-{next(self._ticket_seq)}",
                "summary": summary,
                "severity": severity,
                "status": "open",
                "created_at": datetime.now(UTC).isoformat(),
            }
            self.tickets.append(ticket)
        return {k: v for k, v in ticket.items() if k != "summary"}


TOOL_SPECS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    "search_docs": (SearchDocsIn, SearchDocsOut),
    "get_order_status": (OrderStatusIn, OrderStatusOut),
    "check_warranty": (CheckWarrantyIn, CheckWarrantyOut),
    "recommend_product": (RecommendProductIn, RecommendProductOut),
    "create_support_ticket": (CreateTicketIn, CreateTicketOut),
}

# ------------------------------------------------------------ fault injection


class FaultInjector:
    """Holds per-tool fault modes. Spec format: "get_order_status=timeout;*=empty"."""

    def __init__(self, spec: str = ""):
        self.faults: dict[str, str] = {}
        for item in filter(None, (s.strip() for s in spec.split(";"))):
            name, _, mode = item.partition("=")
            self.set(name.strip(), mode.strip())

    def set(self, tool: str, mode: str | None) -> None:
        if mode is None:
            self.faults.pop(tool, None)
            return
        if mode not in FAULT_MODES:
            raise ValueError(f"Unknown fault mode '{mode}', expected one of {sorted(FAULT_MODES)}")
        self.faults[tool] = mode

    def clear(self) -> None:
        self.faults.clear()

    def get(self, tool: str) -> str | None:
        return self.faults.get(tool, self.faults.get("*"))


# --------------------------------------------------------------------- runner


@dataclass
class ToolRecord:
    name: str
    arguments: dict
    status: str  # ok | invalid_arguments | unknown_tool | timeout | malformed | empty | error
    result: dict | None = None
    error: str | None = None
    latency_ms: int = 0

    def llm_payload(self) -> str:
        if self.status == "ok":
            return json.dumps(self.result, default=str)
        return json.dumps({"error": self.status, "message": self.error})


class ToolRunner:
    _executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")

    def __init__(self, toolbox: Toolbox, settings: Settings):
        self.toolbox = toolbox
        self.timeout_s = settings.tool_timeout_s
        self.faults = FaultInjector(settings.fault_injection)

    def run(self, name: str, raw_arguments: str | dict) -> ToolRecord:
        start = time.perf_counter()
        record = self._run(name, raw_arguments)
        record.latency_ms = int((time.perf_counter() - start) * 1000)
        return record

    def _run(self, name: str, raw_arguments: str | dict) -> ToolRecord:
        try:
            args = json.loads(raw_arguments or "{}") if isinstance(raw_arguments, str) else dict(raw_arguments or {})
        except json.JSONDecodeError:
            return ToolRecord(name, {"_raw": str(raw_arguments)}, "invalid_arguments", error="arguments are not valid JSON")

        if name not in TOOL_SPECS:
            return ToolRecord(name, args, "unknown_tool", error=f"no tool named '{name}'")
        in_model, out_model = TOOL_SPECS[name]
        try:
            parsed = in_model.model_validate(args)
        except ValidationError as exc:
            msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return ToolRecord(name, args, "invalid_arguments", error=msg)

        fault = self.faults.get(name)
        future = self._executor.submit(self._call, name, parsed.model_dump(), fault)
        try:
            raw = future.result(timeout=self.timeout_s)
        except FutureTimeout:
            return ToolRecord(name, args, "timeout", error=f"{name} did not respond within {self.timeout_s:g}s")
        except Exception as exc:  # noqa: BLE001 - any backend failure is reported, not raised
            return ToolRecord(name, args, "error", error=f"{name} failed: {type(exc).__name__}")

        if raw in (None, "", {}, []):
            return ToolRecord(name, args, "empty", error=f"{name} returned no data")
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            result = out_model.model_validate(data).model_dump(mode="json")
        except (json.JSONDecodeError, ValidationError, TypeError):
            return ToolRecord(name, args, "malformed", error=f"{name} returned data that could not be parsed")
        return ToolRecord(name, args, "ok", result=result)

    def _call(self, name: str, kwargs: dict, fault: str | None):
        if fault == "timeout":
            time.sleep(self.timeout_s + 1)
        result = getattr(self.toolbox, name)(**kwargs)
        if fault == "malformed":
            return '{"status": "shipp' + "\x00"  # truncated payload from a broken backend
        if fault == "empty":
            return None
        return result
