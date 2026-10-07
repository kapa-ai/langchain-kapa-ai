.PHONY: format format-check lint typecheck test test-min-deps integration-test integration_test integration_tests build check-wheel eval studio

PYTEST_OFFLINE := --disable-socket --allow-unix-socket
MIN_PYTHON := 3.10
MIN_DEPS_VENV := .venv-min-deps
CHECK_VENV := .venv-check

format:
	uv run ruff format .
	uv run ruff check --fix .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest $(PYTEST_OFFLINE) tests/unit_tests

test-min-deps:
	rm -rf $(MIN_DEPS_VENV)
	uv venv --quiet --python $(MIN_PYTHON) $(MIN_DEPS_VENV)
	uv pip install --quiet --python $(MIN_DEPS_VENV) --resolution lowest-direct --group test -e .
	$(MIN_DEPS_VENV)/bin/python -m pytest $(PYTEST_OFFLINE) tests/unit_tests

integration-test integration_test integration_tests:
	uv run pytest tests/integration_tests

build:
	rm -rf dist
	uv build

check-wheel:
	@for artifact in dist/*.whl dist/*.tar.gz; do \
		echo "Checking $$artifact"; \
		rm -rf $(CHECK_VENV); \
		uv venv --quiet $(CHECK_VENV) || exit 1; \
		uv pip install --quiet --python $(CHECK_VENV) "$$artifact" || exit 1; \
		$(CHECK_VENV)/bin/python scripts/check_install.py || exit 1; \
	done
	rm -rf $(CHECK_VENV)

eval:
	uv run python evals/run.py $(ARGS)

studio:
	uv run langgraph dev --config examples/langgraph.json
