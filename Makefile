# FanCare quality platform. On Windows without make, run the commands after each target directly.
PY ?= python
PROFILE ?= smoke
PROMPTS ?= v1,v2,v3
MODELS ?= groq:openai/gpt-oss-120b,groq:openai/gpt-oss-20b

.PHONY: install run docker lint test test-unit test-api test-web test-data test-safety test-conversation \
        test-tooluse test-extraction eval regress baseline drift load flaky docs-gen promptfoo newman

install:            ## install app + test stack and the Playwright browser
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m playwright install chromium

run:                ## start the API + chat UI on http://localhost:8000
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

docker:             ## run with Docker Compose (add --profile ollama for the local fallback model)
	docker compose up --build

lint:
	$(PY) -m ruff check app eval tests tools monitoring aws

test:               ## everything; LLM suites are skipped without an API key
	$(PY) -m pytest

test-unit:
	$(PY) -m pytest tests/unit tests/extraction -q

test-api:
	$(PY) -m pytest tests/api -q

test-web:
	$(PY) -m pytest tests/web -q

test-data:
	$(PY) -m pytest tests/data -q

test-safety:
	$(PY) -m pytest tests/safety -q

test-conversation:
	$(PY) -m pytest tests/conversation -q

test-tooluse:
	$(PY) -m pytest tests/tooluse -q

test-extraction:
	$(PY) -m eval.extraction

eval:               ## RAG eval (PROFILE=smoke|nojudge|full)
	$(PY) -m eval.run --profile $(PROFILE)

baseline:           ## record the regression baseline from a real run of the reference variant
	$(PY) -m eval.regression --profile $(PROFILE) --prompts v1 --update-baseline

regress:            ## run prompts x models and compare with the baseline (fails on regression)
	$(PY) -m eval.regression --profile $(PROFILE) --prompts $(PROMPTS) --models $(MODELS)

drift:              ## replay the fixed drift sample, update history, chart and drift report
	$(PY) -m monitoring.drift

load:               ## low-concurrency Locust run against a running app
	locust -f tests/load/locustfile.py --host http://localhost:8000 --headless -u 2 -r 1 -t 3m \
		--csv reports/load/run --html reports/load/report.html
	$(PY) -m tests.load.summarize reports/load/run_stats.csv

flaky:              ## rank the flakiest tests from recorded outcomes
	$(PY) -m tools.flaky_report

docs-gen:           ## regenerate the synthetic warranty cards / invoices
	$(PY) -m eval.datasets.documents.generate

promptfoo:          ## promptfoo red team against a running app (needs Node 20+ and GROQ_API_KEY)
	npx promptfoo@0.125.1 redteam run -c redteam/promptfooconfig.yaml

newman:             ## Postman collection against a running app
	npx newman@6.2.3 run postman/FanCare.postman_collection.json -e postman/FanCare.postman_environment.json --working-dir .
