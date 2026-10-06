from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration_tests.live import load_queries, requires_queries

pytestmark = requires_queries

EXAMPLES = Path(__file__).parents[2] / "examples"


def run_example(name: str) -> str:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / name), load_queries()[0]],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return completed.stdout


def test_retrieve_prints_sources() -> None:
    assert "[1] " in run_example("retrieve.py")


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="Set OPENAI_API_KEY to run."
)
def test_answer_prints_evidence_and_answer() -> None:
    output = run_example("answer.py")

    assert "Evidence" in output
    assert "Answer" in output


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="Set OPENAI_API_KEY to run."
)
def test_agent_shows_its_searches() -> None:
    assert "search_knowledge_base" in run_example("agent.py")
