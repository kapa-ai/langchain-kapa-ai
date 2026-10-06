from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urldefrag

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from pydantic import BaseModel, Field

from langchain_kapa_ai import KapaGetDocumentsTool, KapaRetriever

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "evals" / "corpus"
PACKAGES = ["langchain-kapa-ai", "langchain-core", "langchain", "langchain-openai"]
URL_PATTERN = re.compile(r"https?://[^\s)\]>\"'`]+")

ANSWER_INSTRUCTIONS = (
    "Answer the question using only the numbered passages. Cite the passages you "
    "used by their source links. If the passages do not answer the question, say "
    "that the knowledge base does not cover it."
)
AGENT_INSTRUCTIONS = (
    "You answer questions from the knowledge base. Search before you answer. "
    "When a passage is not enough, fetch its whole document with "
    "kapa_get_documents, passing the passage's source link unchanged. Cite the "
    "source links of the passages you used, exactly as the search returned them. "
    "If the knowledge base does not answer the question, say so."
)
GRADER_INSTRUCTIONS = (
    "You grade answers from a retrieval-augmented assistant. You receive the "
    "question, a reference answer, the passages the assistant retrieved, and the "
    "assistant's answer. Judge correctness against the reference answer. Judge "
    "support only against the retrieved passages, never against the reference. "
    "For each passage, say whether it is relevant to the question. When the "
    "reference says the sources do not cover the question, the answer is correct "
    "only if it says so instead of inventing an answer."
)

Category = Literal["direct", "paraphrase", "multi_passage", "ambiguous", "unanswerable"]
Grade = Literal["correct", "partial", "incorrect"]
Support = Literal["supported", "partial", "unsupported"]


class Case(BaseModel):
    id: str
    question: str
    category: Category
    expected_sources: list[str]
    expected_evidence: list[str]
    reference_answer: str


class Grading(BaseModel):
    correctness: Grade = Field(description="How well the answer matches the reference.")
    claim_support: Support = Field(
        description="Whether every claim in the answer is backed by the passages."
    )
    honest_uncertainty: bool = Field(
        description=(
            "True when the answer states its limits where the passages do not "
            "cover the question, or when no such statement is needed."
        )
    )
    relevant_passages: list[int] = Field(
        description="Numbers of the passages that are relevant to the question."
    )
    explanation: str = Field(description="One or two sentences explaining the grades.")


@dataclass
class Attempt:
    passages: list[Document]
    answer: str
    tool_calls: list[dict[str, Any]]
    model_calls: int
    retrieval_seconds: float
    total_seconds: float


def load_cases(path: Path) -> list[Case]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Case.model_validate_json(line) for line in lines if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def same_source(citation: str, expected: str) -> bool:
    candidates = {citation, urldefrag(citation).url}
    return expected.rstrip("/") in {candidate.rstrip("/") for candidate in candidates}


def format_passages(passages: list[Document]) -> str:
    return "\n\n".join(
        f"[{number}] Source: {passage.metadata['source']}\n{passage.page_content}"
        for number, passage in enumerate(passages, start=1)
    )


def make_model(model: str, provider: str) -> BaseChatModel:
    chat_model = init_chat_model(model, model_provider=provider)
    if not isinstance(chat_model, BaseChatModel):
        raise TypeError(f"{provider}:{model} is not a chat model.")
    return chat_model


def run_answer(case: Case, retriever: KapaRetriever, model: BaseChatModel) -> Attempt:
    started = time.perf_counter()
    passages = retriever.invoke(case.question)
    retrieval_seconds = time.perf_counter() - started
    response = model.invoke(
        [
            ("system", ANSWER_INSTRUCTIONS),
            (
                "human",
                f"Passages:\n\n{format_passages(passages)}\n\n"
                f"Question: {case.question}",
            ),
        ]
    )
    return Attempt(
        passages=passages,
        answer=response.text,
        tool_calls=[],
        model_calls=1,
        retrieval_seconds=retrieval_seconds,
        total_seconds=time.perf_counter() - started,
    )


def run_agent(
    case: Case,
    retriever: KapaRetriever,
    documents_tool: KapaGetDocumentsTool,
    model: BaseChatModel,
) -> Attempt:
    search = create_retriever_tool(
        retriever,
        name="search_knowledge_base",
        description=(
            "Search the knowledge base. Returns relevant passages, each with its "
            "source link."
        ),
        document_prompt=PromptTemplate.from_template(
            "Source: {source}\n{page_content}"
        ),
        response_format="content_and_artifact",
    )
    agent = create_agent(
        model, tools=[search, documents_tool], system_prompt=AGENT_INSTRUCTIONS
    )
    started = time.perf_counter()
    state = agent.invoke({"messages": [HumanMessage(case.question)]})
    total_seconds = time.perf_counter() - started
    messages: list[BaseMessage] = state["messages"]

    passages: list[Document] = []
    tool_calls: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            tool_calls += [
                {"name": call["name"], "args": call["args"]}
                for call in message.tool_calls
            ]
        if isinstance(message, ToolMessage) and message.name == search.name:
            passages += list(message.artifact or [])
    final = messages[-1]
    return Attempt(
        passages=passages,
        answer=final.text if isinstance(final, AIMessage) else "",
        tool_calls=tool_calls,
        model_calls=sum(isinstance(message, AIMessage) for message in messages),
        retrieval_seconds=0.0,
        total_seconds=total_seconds,
    )


def grade(case: Case, attempt: Attempt, grader: BaseChatModel) -> Grading:
    structured = grader.with_structured_output(Grading)
    result = structured.invoke(
        [
            ("system", GRADER_INSTRUCTIONS),
            (
                "human",
                f"Question: {case.question}\n\n"
                f"Reference answer: {case.reference_answer}\n\n"
                f"Retrieved passages:\n\n{format_passages(attempt.passages)}\n\n"
                f"Assistant answer:\n{attempt.answer}",
            ),
        ]
    )
    if not isinstance(result, Grading):
        raise TypeError("The grader did not return a grading.")
    return result


def deterministic_metrics(case: Case, attempt: Attempt) -> dict[str, Any]:
    citations = [str(passage.metadata["source"]) for passage in attempt.passages]
    content = normalize(" ".join(passage.page_content for passage in attempt.passages))
    cited = list(dict.fromkeys(URL_PATTERN.findall(attempt.answer)))
    sources_found = [
        expected
        for expected in case.expected_sources
        if any(same_source(citation, expected) for citation in citations)
    ]
    evidence_found = [
        evidence
        for evidence in case.expected_evidence
        if normalize(evidence) in content
    ]
    return {
        "expected_source_recall": (
            len(sources_found) / len(case.expected_sources)
            if case.expected_sources
            else None
        ),
        "expected_evidence_recall": (
            len(evidence_found) / len(case.expected_evidence)
            if case.expected_evidence
            else None
        ),
        "missing_evidence": [
            evidence
            for evidence in case.expected_evidence
            if evidence not in evidence_found
        ],
        "cited_links": cited,
        "citations_from_retrieval": all(
            link.rstrip(".,;") in citations for link in cited
        ),
        "cites_expected_source": any(
            same_source(link.rstrip(".,;"), expected)
            for link in cited
            for expected in case.expected_sources
        ),
        "passage_count": len(attempt.passages),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(values: list[float]) -> float | None:
        return round(statistics.fmean(values), 3) if values else None

    summary: dict[str, Any] = {}
    for mode in sorted({row["mode"] for row in rows}):
        selected = [row for row in rows if row["mode"] == mode]
        metrics = [row["metrics"] for row in selected]
        grades = [row["grading"] for row in selected]
        relevance = [
            len(grading["relevant_passages"]) / metric["passage_count"]
            for metric, grading in zip(metrics, grades, strict=True)
            if metric["passage_count"]
        ]
        summary[mode] = {
            "attempts": len(selected),
            "expected_source_recall": mean(
                [
                    m["expected_source_recall"]
                    for m in metrics
                    if m["expected_source_recall"] is not None
                ]
            ),
            "expected_evidence_recall": mean(
                [
                    m["expected_evidence_recall"]
                    for m in metrics
                    if m["expected_evidence_recall"] is not None
                ]
            ),
            "passage_relevance": mean(relevance),
            "correctness": dict(Counter(grading["correctness"] for grading in grades)),
            "claim_support": dict(
                Counter(grading["claim_support"] for grading in grades)
            ),
            "honest_uncertainty": mean(
                [float(g["honest_uncertainty"]) for g in grades]
            ),
            "citations_from_retrieval": mean(
                [float(m["citations_from_retrieval"]) for m in metrics]
            ),
            "tool_calls": dict(
                Counter(call["name"] for row in selected for call in row["tool_calls"])
            ),
            "mean_model_calls": mean([float(row["model_calls"]) for row in selected]),
            "mean_total_seconds": mean([row["total_seconds"] for row in selected]),
        }
    return summary


def command_run(args: argparse.Namespace) -> None:
    datasets = [Path(path) for path in args.dataset]
    for dataset in datasets:
        if "heldout" in dataset.stem and not args.allow_heldout:
            sys.exit(f"{dataset} is a held-out set; pass --allow-heldout to use it.")

    model = make_model(args.model, args.model_provider)
    grader = make_model(args.grader_model, args.grader_model_provider)
    project = {"project_id": args.project_id} if args.project_id else {}

    output = Path(args.output) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True)
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "system": args.system,
        "modes": args.modes,
        "top_k": args.top_k,
        "max_chars": args.max_chars,
        "repeats": args.repeats,
        "model": {"provider": args.model_provider, "name": args.model},
        "grader": {"provider": args.grader_model_provider, "name": args.grader_model},
        "packages": package_versions(),
        "repository_commit": git("rev-parse", "HEAD"),
        "corpus_revision": git("rev-parse", "HEAD:evals/corpus"),
        "corpus_files": {path.name: sha256(path) for path in sorted(CORPUS.iterdir())},
        "uncommitted_changes": bool(git("status", "--porcelain", "evals", "src")),
        "datasets": {str(path): sha256(path) for path in datasets},
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    rows: list[dict[str, Any]] = []
    with (output / "cases.jsonl").open("w", encoding="utf-8") as cases_file:
        for dataset in datasets:
            for case in load_cases(dataset):
                for mode in args.modes:
                    retriever = KapaRetriever(
                        **project, mode=mode, top_k=args.top_k, max_chars=args.max_chars
                    )
                    for repeat in range(1, args.repeats + 1):
                        if args.system == "agent":
                            documents_tool = KapaGetDocumentsTool(**project)
                            attempt = run_agent(case, retriever, documents_tool, model)
                        else:
                            attempt = run_answer(case, retriever, model)
                        row = {
                            "dataset": str(dataset),
                            "case_id": case.id,
                            "category": case.category,
                            "mode": mode,
                            "repeat": repeat,
                            "question": case.question,
                            "passages": [
                                {
                                    "source": passage.metadata["source"],
                                    "content": passage.page_content,
                                }
                                for passage in attempt.passages
                            ],
                            "answer": attempt.answer,
                            "tool_calls": attempt.tool_calls,
                            "model_calls": attempt.model_calls,
                            "retrieval_seconds": round(attempt.retrieval_seconds, 3),
                            "total_seconds": round(attempt.total_seconds, 3),
                            "metrics": deterministic_metrics(case, attempt),
                            "grading": grade(case, attempt, grader).model_dump(),
                        }
                        rows.append(row)
                        cases_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        print(
                            f"{case.id} {mode} #{repeat}: "
                            f"{row['grading']['correctness']}, "
                            f"evidence {row['metrics']['expected_evidence_recall']}"
                        )

    summary = summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Results written to {output}")


def command_calibrate(args: argparse.Namespace) -> None:
    rows = [
        json.loads(line)
        for line in (Path(args.run) / "cases.jsonl").read_text().splitlines()
        if line.strip()
    ]
    graded = {
        (row["case_id"], row["mode"], row["repeat"]): row["grading"] for row in rows
    }
    judgments = [
        json.loads(line)
        for line in Path(args.judgments).read_text().splitlines()
        if line.strip()
    ]
    for label in ("correctness", "claim_support", "honest_uncertainty"):
        pairs = [
            (judgment[label], graded[key][label])
            for judgment in judgments
            if label in judgment
            and (key := (judgment["case_id"], judgment["mode"], judgment["repeat"]))
            in graded
        ]
        if not pairs:
            continue
        agreement = sum(human == model for human, model in pairs) / len(pairs)
        print(f"{label}: {agreement:.0%} agreement over {len(pairs)} judgments")
        for (human, model), count in sorted(Counter(pairs).items(), key=str):
            if human != model:
                print(f"  human {human!r}, grader {model!r}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and calibrate retrieval and answer evaluations."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run datasets and grade the answers.")
    run.add_argument("--dataset", action="append", required=True)
    run.add_argument("--system", choices=["answer", "agent"], default="answer")
    run.add_argument(
        "--modes", nargs="+", choices=["default", "deep"], default=["default", "deep"]
    )
    run.add_argument("--top-k", type=int, default=None)
    run.add_argument("--max-chars", type=int, default=None)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--project-id", default=None)
    run.add_argument("--model", default=os.environ.get("KAPA_EXAMPLE_MODEL", "gpt-5.1"))
    run.add_argument(
        "--model-provider",
        default=os.environ.get("KAPA_EXAMPLE_MODEL_PROVIDER", "openai"),
    )
    run.add_argument("--grader-model", default="gpt-5.1")
    run.add_argument("--grader-model-provider", default="openai")
    run.add_argument("--allow-heldout", action="store_true")
    run.add_argument("--output", default=str(ROOT / "evals" / "results"))
    run.set_defaults(handler=command_run)

    calibrate = commands.add_parser(
        "calibrate", help="Compare grader labels with your own judgments."
    )
    calibrate.add_argument("--run", required=True)
    calibrate.add_argument("--judgments", required=True)
    calibrate.set_defaults(handler=command_calibrate)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
