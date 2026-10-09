from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path

import pytest

QUERIES_FILE = Path(__file__).with_name("queries.local.txt")


def credentials_present() -> bool:
    return bool(os.environ.get("KAPA_API_KEY") and os.environ.get("KAPA_PROJECT_ID"))


def load_queries() -> list[str]:
    text = os.environ.get("KAPA_TEST_QUERIES")
    if text is None and QUERIES_FILE.exists():
        text = QUERIES_FILE.read_text(encoding="utf-8")
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


requires_credentials = pytest.mark.skipif(
    not credentials_present(),
    reason="Set KAPA_API_KEY and KAPA_PROJECT_ID to run live tests.",
)

requires_queries = pytest.mark.skipif(
    not credentials_present() or not load_queries(),
    reason=(
        "Set KAPA_API_KEY and KAPA_PROJECT_ID, and provide queries in "
        "KAPA_TEST_QUERIES or tests/integration_tests/queries.local.txt."
    ),
)


def model_skip_reason() -> str | None:
    model = os.environ.get("KAPA_EXAMPLE_MODEL", "")
    if ":" not in model:
        return 'Set KAPA_EXAMPLE_MODEL to "<provider>:<model>" to run.'
    provider = model.split(":", 1)[0]
    if find_spec("langchain_" + provider.replace("-", "_")) is None:
        package = "langchain-" + provider.replace("_", "-")
        return (
            f"Install {package} for the project interpreter, for example with "
            f"`uv run --with {package} pytest tests/integration_tests`."
        )
    return None


requires_model = pytest.mark.skipif(
    model_skip_reason() is not None, reason=model_skip_reason() or ""
)
