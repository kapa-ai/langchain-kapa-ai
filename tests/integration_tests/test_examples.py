from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.integration_tests.live import load_queries, requires_model, requires_queries

pytestmark = requires_queries

EXAMPLES = Path(__file__).parents[2] / "examples"


def run_example(name: str) -> str:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / name), load_queries()[0]],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    stderr = "\n".join(completed.stderr.splitlines()[-40:])
    assert completed.returncode == 0, (
        f"{name} exited with status {completed.returncode}:\n{stderr}"
    )
    return completed.stdout


def test_retrieve_prints_sources() -> None:
    assert "[1] " in run_example("retrieve.py")


@requires_model
def test_answer_prints_evidence_and_answer() -> None:
    output = run_example("answer.py")

    assert "Evidence" in output
    assert "Answer" in output


@requires_model
def test_agent_shows_its_searches() -> None:
    assert "search_knowledge_sources" in run_example("agent.py")
