.DEFAULT_GOAL := help
PY ?= python
VENV := .venv
BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
BIN := $(VENV)/Scripts
endif

.PHONY: help dev lint types test test-unit test-contract test-integration test-e2e \
        test-adversarial check up down seed eval clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

dev: ## Create the venv and install everything needed to develop
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[api,storage,dev]"

lint: ## Ruff + the import-linter dependency contracts
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests
	$(BIN)/lint-imports --config importlinter.ini

types: ## Strict mypy over the package
	$(BIN)/mypy src

# Unit tests must stay fast enough to run on save: no network, no database.
test-unit: ## Unit tests only
	$(BIN)/pytest tests/unit -q

test-contract: ## Conformance suite, parameterized over every protocol implementation
	$(BIN)/pytest tests/contract -q

test-integration: ## Against the docker compose stack
	$(BIN)/pytest tests/integration -q

test-e2e: ## Full graph against the seeded corpus
	$(BIN)/pytest tests/e2e -q

test-adversarial: ## Injection, poisoning, ACL probes. Blocking in CI
	$(BIN)/pytest tests/adversarial -q

test: ## Everything that does not need cloud credentials
	$(BIN)/pytest tests -q

check: lint types test-unit test-contract ## What CI runs before the slow suites

up: ## Bring up the local stack (api only)
	docker compose up -d

up-full: ## Bring up the api plus Postgres, Redis and Qdrant
	docker compose --profile full up -d

serve: ## Run the API directly, without docker
	$(BIN)/uvicorn prag.api.http.main:app --reload --port 8080

down: ## Tear down the local stack
	docker compose down -v

seed: ## Seed the local corpus and prove one query answers end to end
	$(BIN)/python scripts/seed_corpus.py

eval: ## Run the evaluation suite and print the scorecard
	$(BIN)/python scripts/run_eval.py

clean: ## Remove caches and build artifacts
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
