from __future__ import annotations

import os
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
