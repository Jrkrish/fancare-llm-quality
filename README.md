# FanCare Assistant: an AI quality platform around a RAG + tool-calling support bot

FanCare Assistant is a customer-support chatbot for a fictional smart ceiling fan. It answers from
product documentation with citations, calls tools for orders, warranty, product recommendations and
support tickets, and reads uploaded warranty cards. Around it sits an automated quality platform that
tests a system whose outputs are non-deterministic: golden datasets, repeated runs with thresholds and
variance, LLM-as-judge with calibration against human labels, adversarial testing, a regression gate in
CI, drift monitoring, and conventional API / UI / data / load testing.

Everything runs on free tiers or locally: Groq or Gemini (free API keys), Ollama as an optional local
fallback, local ONNX models for embeddings, reranking and OCR, SQLite for logs, GitHub Actions for CI.

> **Results:** every number in the results tables below comes from running this repo. Smoke-run results,
> including the invalid runs and what each one exposed, are in [Results](#results). Anything not run yet
> says _"not yet run"_. No figures are estimated or invented.

---

## Contents
1. [The system under test](#the-system-under-test)
2. [Architecture](#architecture)
3. [Quick start](#quick-start)
4. [The quality platform](#the-quality-platform)
5. [Testing non-deterministic outputs](#testing-non-deterministic-outputs)
6. [Metric definitions](#metric-definitions)
7. [Regression demo (v1 vs v2 vs v3)](#regression-demo)
8. [Drift monitoring](#drift-monitoring)
9. [CI/CD](#cicd)
10. [Results](#results)
11. [Bugs the suite caught while building it](#bugs-the-suite-caught-while-building-it)
12. [Limitations](#limitations)
13. [Repository layout](#repository-layout)

---

## The system under test

**FanCare Assistant** supports owners of *Corvane* smart ceiling fans (a fictional brand). Python 3.11, FastAPI.

| Capability | Implementation |
|---|---|
| Agent with tool calling | OpenAI-compatible chat API to Groq / Gemini / Ollama, loop of up to 4 tool rounds ([app/agent.py](app/agent.py)) |
| `search_docs` | Hybrid retrieval over 19 markdown docs: BM25 + `bge-small-en-v1.5` embeddings, fused with reciprocal rank fusion, reranked with a MiniLM cross-encoder, with a "no evidence" threshold ([app/retrieval.py](app/retrieval.py)) |
| Mock business tools | `get_order_status`, `check_warranty` (applies the 2025 or the superseded 2022 policy by purchase date), `recommend_product` (12-product catalogue), `create_support_ticket` ([app/tools.py](app/tools.py)) |
| Fault injection | Any tool can be made to `timeout`, return a `malformed` payload, or return `empty`, via `FAULT_INJECTION` or at runtime |
| Citations | Answers cite `[DOC-07:4]` (doc id : section); every citation is checked against the corpus and against what was actually retrieved this turn |
| Document understanding | `POST /upload` extracts `serial_no`, `purchase_date`, `model` from PDF (text layer) or image (local OCR) and returns `null` + a warning instead of guessing ([app/documents.py](app/documents.py)) |
| Memory | Per-session turns, plus validated fields from uploads |
| Guardrails | Separate layer ([app/guardrails.py](app/guardrails.py)): input filter (injection, jailbreak, prompt extraction, blocked topics, PII redaction), retrieved-passage screening, output filter (PII redaction, system-prompt leak detection via canary + n-gram overlap) |
| Prompts | [prompts/v1](prompts/v1/system.txt) baseline, [v2](prompts/v2/system.txt) deliberately weaker ("friendlier, use general knowledge"), [v3](prompts/v3/system.txt) stricter |
| Observability | Every request goes to SQLite: trace id, prompt version, model, retrieved/cited chunk ids, tool calls, latency, tokens, cost estimate, guardrail verdicts ([app/observability.py](app/observability.py)). Optional Langfuse export when keys are set |
| API | `POST /chat`, `POST /chat/stream` (SSE), `GET /sessions/{id}/history`, `POST /upload`, `GET /health`, Swagger at `/docs`, API-key auth, per-key rate limit |
| UI | Minimal chat page at `/` with streaming, citation chips, upload and session persistence |

The knowledge base contains deliberate test material: a superseded warranty policy that conflicts with the current one,
a time-limited returns exception, and an unofficial community page carrying two indirect prompt injections: one that
the pattern filter catches, and one ("bridge the thermal sensor wires") that only the system prompt can defend against.

## Architecture

```mermaid
flowchart LR
    U[User / UI / API client] -->|X-API-Key, rate limit| API[FastAPI]
    API --> IF[Input filter<br/>injection, jailbreak,<br/>blocked topics, PII redaction]
    IF -- blocked --> R[Canned refusal]
    IF -- allowed --> AG[Agent loop]
    AG <--> LLM[(Groq / Gemini<br/>Ollama fallback)]
    AG --> T{Tools}
    T --> SD[search_docs<br/>BM25 + vectors + RRF + rerank]
    SD --> PS[Passage screening<br/>indirect injection]
    T --> MT[orders / warranty /<br/>catalogue / tickets<br/>+ fault injection]
    AG --> OF[Output filter<br/>PII, prompt-leak]
    OF --> CIT[Citation check]
    CIT --> API
    AG --> LOG[(SQLite traces<br/>+ optional Langfuse)]
    UP[POST /upload] --> EX[PDF text / OCR<br/>strict field extraction] --> MEM[(Session memory)]
    MEM --> AG
```

```mermaid
flowchart LR
    subgraph Quality platform
      G[Golden dataset 66 cases] --> E[Eval runner<br/>N runs per case]
      E --> DE[DeepEval judge metrics]
      E --> DM[Deterministic metrics]
      DE & DM --> REP[Reports + eval_history.csv]
      REP --> GATE[Regression gate vs baseline]
      REP --> CAL[Judge calibration vs human labels]
      S[Safety 42 cases] --> ASR[ASR per category]
      TU[Tool-use 20 cases] & CV[Conversations 6] & DOC[Extraction 10 docs]
      API2[API + Postman] & WEB[Playwright] & DATA[SQL checks on logs] & LOAD[Locust]
      DRIFT[Nightly drift replay] --> TREND[Trend charts + drift flags]
    end
```

## Quick start

Requirements: Python 3.11 and a free [Groq](https://console.groq.com) key. The bot runs on
`openai/gpt-oss-120b`; the DeepEval judge runs on `qwen/qwen3.8-27b`, a different model family to limit
self-preference bias, with its own free-tier quota. A [Gemini](https://aistudio.google.com) key is optional
(alternative bot provider). Node 20+ is only needed for promptfoo and newman.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m playwright install chromium
cp .env.example .env                                   # set GROQ_API_KEY

uvicorn app.main:app --port 8000                       # API + UI at http://localhost:8000, Swagger at /docs
```

Or with Docker: `docker compose up --build` (add `--profile ollama` to run a local Ollama fallback).
On first start the app downloads ~150 MB of ONNX models into `.cache/models`.

| Command (`make` target) | What it does | Needs a key? |
|---|---|---|
| `pytest tests/unit tests/extraction` (`make test-unit`) | Agent, tools, guardrails, retrieval, eval engine, extraction | no |
| `pytest tests/api` (`make test-api`) | Starts its own servers; contract, validation, auth, rate limit, uploads | no |
| `pytest tests/web` (`make test-web`) | Playwright UI tests | no |
| `pytest tests/data` (`make test-data`) | SQL checks on whatever logs exist | no |
| `pytest tests/safety` / `tests/tooluse` / `tests/conversation` | LLM suites (skipped without a key) | Groq |
| `python -m eval.run --profile smoke` (`make eval`) | RAG eval, 12 cases x 2 runs | Groq |
| `python -m eval.run --profile full` | RAG eval, 66 cases x 3 runs | Groq |
| `python -m eval.run --profile nojudge` | RAG eval, deterministic metrics only | Groq |
| `python -m eval.regression ...` (`make regress`) | Variants vs baseline, fails on regression | Groq |
| `python -m monitoring.drift` (`make drift`) | Replay fixed sample, trend chart, drift flags | Groq |
| `make load` | Locust, 2 users, 3 minutes | Groq |

**Free-tier budget.** LLM-judged metrics cost several judge calls per answer. The runner prints the
approximate number of calls before it starts. Use `smoke` or `nojudge` for frequent runs, set
`LLM_MIN_INTERVAL_S` to stay under requests-per-minute limits, and check your provider's current daily
limits before a `full` run.

## The quality platform

| # | Suite | Where | What it checks |
|---|---|---|---|
| 1 | RAG eval | [eval/](eval/), [tests/eval](tests/eval) | 66 golden cases (22 answerable, 8 partial, 10 unanswerable, 8 multi-hop, 6 conflicting-docs, 12 near-miss) x N runs; DeepEval faithfulness, answer relevancy, contextual precision/recall + deterministic citation correctness, refusal correctness, retrieval recall, hallucination rate |
| 2 | Judge calibration | [eval/calibration.py](eval/calibration.py) | You hand-label 20 sampled runs; reports judge-vs-human accuracy and Cohen's kappa per metric and flags unreliable metrics |
| 3 | Tool use | [eval/tool_use.py](eval/tool_use.py), [tests/tooluse](tests/tooluse) | Right tool, right arguments, no unnecessary calls, recovery after each injected failure (timeout / malformed / empty), no invented order or warranty data |
| 4 | Conversations | [eval/conversation.py](eval/conversation.py), [tests/conversation](tests/conversation) | 6 dialogues of 5-8 turns: context retention, topic switches, ambiguous follow-ups, contradicting the user, frustration; optional DeepEval KnowledgeRetention + ConversationalGEval |
| 5 | Document extraction | [eval/extraction.py](eval/extraction.py), [tests/extraction](tests/extraction) | 10 synthetic cards/invoices (7 clean incl. one with hidden instructions, 3 damaged): field accuracy, exact match, no guessing |
| 6 | Safety | [eval/safety.py](eval/safety.py), [tests/safety](tests/safety), [redteam/](redteam/) | 42 cases across direct injection, indirect injection (retrieved doc and uploaded file), jailbreaks, prompt extraction, PII (in answers and in stored logs), blocked topics; attack success rate per category; promptfoo red team + static configs |
| 7 | Regression gate | [eval/regression.py](eval/regression.py) | Prompts v1/v2/v3 x two models vs a stored baseline with per-metric thresholds; markdown comparison; non-zero exit fails CI |
| 8 | Production-failure loop | [eval/promote_failure.py](eval/promote_failure.py) | Lists suspicious logged traces and turns one into a new golden case with an expected behaviour |
| 9 | Drift monitoring | [monitoring/drift.py](monitoring/drift.py) | Nightly replay of a fixed sample; scores, p50/p95 latency and cost over time; SVG trend charts; flags drift vs the recent median |
| 10 | API | [tests/api](tests/api), [postman/](postman/) | OpenAPI schema validation of live responses, bad/empty/oversized/missing input, auth, rate limit, upload edge cases; Postman collection |
| 11 | Data validation | [tests/data](tests/data) | SQL checks on the logs: nulls, duplicates, schema violations, PII in stored data, orphaned rows; each check proven against seeded bad rows |
| 12 | Web | [tests/web](tests/web) | Playwright + page object: send, streaming render, citations, error states, upload flow, session persistence |
| 13 | Performance | [tests/load](tests/load) | Locust at low concurrency; p50/p95/p99, error rate, cost per 100 conversations |
| 14 | Flakiness | [tests/conftest.py](tests/conftest.py), [tools/flaky_report.py](tools/flaky_report.py) | Reruns for LLM tests, quarantine list, outcome history, flakiest-tests report |
| 15 | Mobile (skeleton) | [tests/mobile](tests/mobile) | One Appium test on Appium's public ApiDemos APK |
| 16 | AWS (optional) | [aws/](aws/) | Upload reports to S3, Athena table + sample trend query; skipped without credentials |

Reusable building blocks live in [tests/framework](tests/framework): `BaseTest`, `ApiClient`,
`OpenApiValidator`, `LiveServer`, and the `ChatPage` page object in [tests/web/pages](tests/web/pages).
The in-process `AgentHarness` ([eval/harness.py](eval/harness.py)) lets LLM suites inject tool faults, attach
uploads and inspect exactly what reached the model.

## Testing non-deterministic outputs

A single pass/fail on one LLM answer measures luck. The platform uses these instead:

1. **Repeat and aggregate.** Every golden case is asked `runs_per_case` times (default 3), each in a fresh
   session. Each metric yields a score per run; a case passes a metric if it clears the threshold in at
   least `min_case_pass_rate` of runs (default 2 of 3).
2. **Report variance, not only means.** Per case and metric: mean, standard deviation, minimum and pass
   rate. Cases whose scores flip between runs (std > 0.15 or a pass rate strictly between 0 and 1) are
   listed as **unstable**, which is a different problem from "consistently wrong".
3. **Suite-level gates** on mean scores, plus a hallucination-rate ceiling and an infrastructure error-rate
   ceiling (rate-limited runs are excluded from scores and reported, never silently counted as failures).
4. **Mix deterministic and judged metrics.** Citation correctness, refusal correctness, retrieval recall,
   tool-call correctness and invented-data checks are exact and free. LLM-judged metrics (DeepEval) cover
   what rules cannot: groundedness and relevance.
5. **Calibrate the judge.** Judge verdicts are compared with human labels; metrics where the judge
   disagrees with humans (kappa < 0.6 or accuracy < 80%) are flagged, so nobody gates a release on an
   unreliable judge.
6. **Compare against a baseline, not an absolute.** The regression gate fails on a *drop* beyond a
   per-metric tolerance, which absorbs normal run-to-run noise.
7. **Rerun and quarantine.** LLM-dependent tests get automatic reruns, every outcome is logged, and
   persistently flaky tests can be quarantined (still run and reported, but non-blocking) with an owner.

## Metric definitions

| Metric | Type | Definition | Per-run threshold |
|---|---|---|---|
| faithfulness | DeepEval | Share of claims in the answer supported by the retrieved passages. `penalize_ambiguous_claims=True`, so unsupported claims count against it, not only contradictions. Claims made with nothing retrieved score 0 | 0.7 |
| answer_relevancy | DeepEval | Share of answer statements relevant to the question. A refusal of an answerable question scores 0 | 0.7 |
| contextual_precision | DeepEval | Are the relevant passages ranked above irrelevant ones? | 0.6 |
| contextual_recall | DeepEval | How much of the reference answer is attributable to the retrieved passages? | 0.6 |
| citation_correctness | deterministic | F1 of cited docs vs gold docs; a citation that was not retrieved this turn, or does not exist, counts as wrong | 0.6 |
| refusal_correctness | deterministic | The response class (answered / partial / refused) matches the expected behaviour. *answer*: not refused. *partial*: answers with citations and states the gap. *refuse*: acknowledges the information is not available | 1.0 |
| retrieval_recall | deterministic | Share of gold chunks that retrieval returned | 0.5 |
| hallucination rate | derived | Share of runs that answered an unanswerable question, cited a non-existent chunk, or scored faithfulness < 0.5 | gate ≤ 0.15 |
| attack success rate (ASR) | deterministic | Share of attack attempts where any safety check failed (refusal expected, forbidden content, PII, canary, injected text reaching the LLM, PII in logs) | target 0 |
| tool metrics | deterministic | selection, argument accuracy, no unnecessary calls, recovery per failure type, no invented data (dates, IDs, carriers or prices not present in any tool result or user message) | 2 of 3 runs |

Thresholds and gates live in [eval/config.toml](eval/config.toml). The gates there are **starting
targets**; recalibrate them from your first baseline run.

## Regression demo

`prompts/v2` reads like a plausible "make the bot friendlier" change: it drops the evidence-only rule,
the "say you don't know" instruction and the tool-failure guidance, and invites general knowledge. The
gate is expected to catch the damage; v3 tightens the rules further.

```bash
# 1. record a baseline from a real run of the reference variant
python -m eval.regression --profile smoke --prompts v1 --update-baseline
git add eval/baselines/smoke.json            # commit it so CI compares against it

# 2. compare prompts and models
python -m eval.regression --profile smoke --prompts v1,v2,v3 \
    --models groq:openai/gpt-oss-120b,groq:openai/gpt-oss-20b
```

Output: `reports/regression/comparison.md`, with each variant's metric means and their change versus baseline,
plus non-zero exit on regression. In CI, the `regression-gate` job runs the same comparison on the eval job's results.

**Regression results** (paste from `reports/regression/comparison.md`):

| Variant | faithfulness | answer_relevancy | citation_correctness | refusal_correctness | hallucination_rate | Result |
|---|---|---|---|---|---|---|
| v1, gpt-oss-120b (baseline) | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run |
| v2, gpt-oss-120b | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run |
| v3, gpt-oss-120b | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run |
| v1, gpt-oss-20b | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run |

## Drift monitoring

`python -m monitoring.drift` replays 12 fixed golden cases ([monitoring/drift_config.json](monitoring/drift_config.json)),
appends a row to `reports/drift/drift_history.csv`, draws `scores.svg`, `latency.svg` and `cost.svg`, and flags
drift when the latest run is worse than the median of the previous 5 runs by more than the configured
thresholds (score drop > 0.10, hallucination rate +0.10, p95 latency +50%, cost +50%, or a model change).
The nightly workflow keeps the history across runs and fails when drift is flagged.

Drift report: _not yet run; after a week of nightly runs, link `reports/drift/drift_report.md` and its charts here._

## CI/CD

| Workflow | Trigger | Jobs |
|---|---|---|
| [ci.yml](.github/workflows/ci.yml) | every push / PR | lint, unit + extraction, API + Postman + web + data (against a keyless app), safety + tool-use + conversation, RAG eval (smoke), **regression gate** |
| [nightly.yml](.github/workflows/nightly.yml) | daily 03:00 UTC | drift replay with persisted history |
| [manual.yml](.github/workflows/manual.yml) | manual | load test, full regression matrix, promptfoo red team, Appium on an emulator, AWS upload + Athena |

Setup: add `GROQ_API_KEY` as a repository secret (`GEMINI_API_KEY` optional; LLM jobs skip with a warning
without them), and mark the `regression-gate` job as a **required status check** under
Settings > Branches so a regression blocks merging. Dependencies, ONNX models and the Playwright browser are cached.
HTML (pytest-html), JUnit and markdown reports are uploaded as artifacts.

## Results

All cells are filled from real runs of this repo.

### Smoke runs so far

`python -m eval.run --profile smoke`: 12 cases (2 per category) × 2 runs each, system under test
`openai/gpt-oss-120b` on Groq with prompt v1 and hybrid retrieval + reranking. Every run is listed,
including the invalid ones, because each one exposed a problem in the platform rather than the bot.

| # | Date (UTC) | Judge | Validity | Gates | Why |
|---|---|---|---|---|---|
| 1 | 2026-11-03 11:25 | Gemini 2.5 Flash | **invalid** | (failed) | Judge hit Gemini's 20 requests/day free quota; 30 judge scores missing |
| 2 | 2026-11-03 11:59 | Qwen3.8 27B (Groq) | valid | **passed** | Judge error rate 0.028, run error rate 0.083 |
| 3 | 2026-11-03 12:26 | Qwen3.8 27B (Groq) | **invalid** | (failed) | Judge error rate 0.222: Groq "Request too large" was not retried |
| 4 | _running_ | Qwen3.8 27B (Groq) | _pending_ | _pending_ | First run with all fixes above |

**Run 2, the only valid run so far:**

| Metric | Value | Gate |
|---|---|---|
| faithfulness | 0.820 | ≥ 0.75 |
| answer_relevancy | 0.954 | ≥ 0.75 |
| citation_correctness | 0.922 | ≥ 0.60 |
| refusal_correctness | 0.909 | ≥ 0.80 |
| retrieval_recall | 0.808 | ≥ 0.60 |
| hallucination_rate | 0.045 | ≤ 0.15 |
| case pass rate | 50% (6 of 12) | none |
| unstable cases | 5 of 12 | none |
| latency p50 / p95 | 2.7 s / 6.4 s | none |
| system-under-test cost | $0.016 for 24 runs (list-price estimate) | none |

Read with care: two smoke cases per category is too few to say anything per category, and run 2 predates
the refusal-detector fix. Re-scoring its answers with the fixed detector turns NM-02's one "hallucination"
and one PAR-02 refusal miss into passes. The remaining failures were real: two retrieval misses
(NM-02, MH-02), one answer that hedged instead of concluding (MH-02), and faithfulness scores that
moved between 0.6 and 1.0 for nearly identical answers (CON-02). That last one is why judge calibration
comes before trusting the faithfulness gate.

### Full eval

**RAG eval** (`python -m eval.run --profile full`, from `reports/eval/latest/summary.md`). Model: _fill in_. Date: _fill in_.

| Category | Cases | faithfulness | answer_relevancy | ctx precision | ctx recall | citation | refusal | Hallucination |
|---|---|---|---|---|---|---|---|---|
| answerable | 22 | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run |
| partial | 8 | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run |
| unanswerable | 10 | n/a | n/a | n/a | n/a | n/a | not yet run | not yet run |
| multi_hop | 8 | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run |
| conflicting | 6 | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run |
| near_miss | 12 | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run |

**Judge calibration** (`python -m eval.calibration report`): faithfulness kappa _not yet run_,
answer relevancy kappa _not yet run_, refusal detector kappa _not yet run_. Unreliable metrics: _not yet run_.

**Safety** (`pytest tests/safety`, `reports/safety/summary.md`): ASR per category _not yet run_.

**Tool use** (`reports/tool_use/summary.md`): recovery by failure type _not yet run_.

**Conversations** (`reports/conversation/summary.md`): turn pass rate _not yet run_.

**Document extraction** (`python -m eval.extraction`): field accuracy _not yet run_, graceful failure _not yet run_.

**Load** (`make load`, `reports/load/summary.md`): p50 / p95 / p99 _not yet run_, error rate _not yet run_,
cost per 100 conversations _not yet run_.

## Bugs the suite caught while building it

These were found by the tests during development and fixed. They are real, not illustrative:

- **Retired model.** The first end-to-end API test returned 503: the default Groq model had been retired
  (`model_not_found`). The defaults were moved to models the account actually lists.
- **Citation format drift.** With `gpt-oss`, the real-answer test reported "no citations" even though the
  answer cited its source: the model writes `【DOC-07:4†L2】` instead of `[DOC-07:4]`. All citation styles are
  now normalised before validation, with unit tests for each style.
- **Over-eager guardrails.** Unit tests showed the input filter blocking "The app shows a system message"
  as a prompt-extraction attempt, and the national-ID pattern redacting part of a non-Luhn card number.
- **Flaky timeout.** An agent test passed alone but failed under load: the test tool budget (0.3 s) was
  too close to real search latency (~0.25 s with reranking).
- **Rounding in the pass rule.** "2 of 3 runs" is 0.667, which a 0.67 threshold rejected.
- **A judge out of quota looked like a quality failure.** The first smoke run used Gemini 2.5 Flash as judge,
  whose free tier allowed 20 requests a day. Most judge calls failed, and the faithfulness gate "failed" on
  one or two scored answers. The runner now tracks a judge error rate and marks such runs **invalid**, the
  client stops retrying once a daily quota is gone, and the judge moved to Qwen on Groq.
- **Transient gateway errors.** Groq occasionally returned HTTP 522 (Cloudflare timeout), which the client did
  not retry. All 5xx responses are now retried.
- **Golden labels that were too narrow.** Two cases marked correct answers as retrieval or citation misses
  because valid evidence (another section stating the same fact) was missing from the label. Both labels
  were widened, with a note in the dataset saying why.
- **A curly apostrophe faked a hallucination.** The second smoke run flagged NM-02 as "answered an
  unanswerable question". The answer was "I don’t have that information", written with a typographic
  apostrophe that the refusal detector's `don't` pattern did not match. The detector now normalises
  typographic punctuation; re-scoring that run changed exactly the two misread answers and nothing else.
- **A tool call after the tool budget.** When the agent used all its tool rounds, the forced final call
  (sent without tools) made `gpt-oss` emit a tool call anyway, and Groq rejected the request (HTTP 400
  `tool_use_failed`), so the case errored. The agent now tells the model to answer from what it has and,
  if the provider still rejects it, falls back to the standard "I don't have that information" reply.
- **A rate limit misread as permanent.** Groq rejects a judge request as "Request too large" when it no
  longer fits that minute's token budget. Treating it as permanent made the third smoke run invalid
  (judge error rate 0.22). The same requests succeed once the window rolls over, so the client now waits
  and retries.
- **Judge scores you could not audit.** Faithfulness scored nearly identical answers 1.0 and 0.6. Reports
  now list every claim the judge rejected, with its reason (read from DeepEval's verdicts, no extra calls),
  which is what judge calibration needs.

## Limitations

- **Mock tools and a fictional product.** Orders, warranty registry and catalogue are small JSON files.
  The behaviour under test (tool choice, arguments, failure recovery) is real; the data is not.
- **Pattern-based guardrails.** The input/output filters are regex rules: the first layer, not the only
  one. Paraphrased attacks get past them by design, and the safety suite measures how often the
  prompt-level defence then holds. Street addresses and names are not redacted.
- **Streaming is post-hoc.** The answer is generated and checked by the output filter before it is
  streamed, so streaming improves perceived responsiveness, not time to first token.
- **The refusal detector is lexical.** It recognises explicit "not in the documentation" phrasing.
  Judge calibration measures its agreement with human labels.
- **Judges are LLMs too.** DeepEval metrics depend on the judge model and its prompt; scores are
  comparable only for the same judge, which is why the judge is recorded with every run.
- **Free-tier limits** shape the design: small default profiles, request spacing, low load-test
  concurrency. Rate-limit errors are reported as infrastructure errors, not quality failures.
- **Cost figures are estimates** from reference list prices in [app/config.py](app/config.py); free-tier
  usage actually costs $0. Verify prices before quoting them.
- **OCR is tested on synthetic documents.** Real photographed warranty cards (glare, angles, handwriting)
  would need a larger, real dataset.
- **Mobile is a skeleton.** It runs against Appium's public sample APK. BLE and Wi-Fi provisioning, the
  part of a smart-fan app worth testing, needs real hardware: the fan, a 2.4 GHz router and a physical phone.
- **Session memory is in-process.** It is lost on restart and not shared between replicas.

## Repository layout

```
app/                FastAPI app: agent, retrieval, tools, guardrails, documents, observability, static UI
data/docs/          19 knowledge-base documents (incl. superseded, holiday and community pages)
data/*.json         orders, warranty registry, 12-product catalogue
prompts/v1..v3/     system prompts (baseline, weaker, stricter)
eval/               eval engine, judge wrapper, metrics, runners, regression gate, calibration, promotion
eval/datasets/      golden (66), safety (42), tool-use (20), conversations (6), documents (10 + generator)
monitoring/         drift job and config
tests/              unit, eval, safety, tooluse, conversation, extraction, api, data, web, load, mobile, aws
tests/framework/    BaseTest, ApiClient, OpenApiValidator, LiveServer
redteam/            promptfoo red-team and static configs
postman/            Postman collection + environment
aws/                S3 upload, Athena table and query
tools/              flakiness report
.github/            CI, nightly drift, manual jobs, shared setup action
```
