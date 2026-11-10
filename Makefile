.PHONY: clean build publish publish-test install dev lint format format-check isort-check typecheck check

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info
	find src -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find src -type f -name "*.pyc" -delete

build: clean
	uv build

publish: clean build
	@set -a && . ./.env && set +a && uv publish

publish-test: clean build
	@set -a && . ./.env && set +a && \
		UV_PUBLISH_TOKEN="$$UV_PUBLISH_TOKEN_TESTPYPI" uv publish \
		--publish-url https://test.pypi.org/legacy/

install:
	uv pip install -e .

dev:
	uv sync --dev

lint:
	uv run ruff check src/

format:
	uv run isort src/
	uv run ruff format src/

format-check:
	uv run ruff format --check src/

isort-check:
	uv run isort --check-only --diff src/

typecheck:
	uv run mypy

check: lint format-check isort-check typecheck
