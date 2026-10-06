from __future__ import annotations

import re
from collections import Counter
from urllib.parse import urldefrag

import pytest

from langchain_kapa_ai import (
    KapaDocumentRequestResult,
    KapaDocumentsPage,
    KapaGetDocumentsTool,
    KapaRetriever,
)
from tests.integration_tests.live import load_queries, requires_queries

pytestmark = requires_queries


def anchor_kind(result: KapaDocumentRequestResult) -> str:
    link = result.requested_url or ""
    fragment = urldefrag(link).fragment
    if "#" not in link:
        return "no anchor"
    if result.match == "exact":
        return "exact fragment-bearing link"
    if re.fullmatch(r"L\d+(-L\d+)?", fragment):
        return "line"
    if re.fullmatch(r"page=\d+", fragment):
        return "page"
    return "heading"


def resolve(tool: KapaGetDocumentsTool, links: list[str]) -> list[KapaDocumentsPage]:
    pages: list[KapaDocumentsPage] = []
    page: int | None = 1
    while page is not None:
        message = tool.invoke(
            {
                "name": tool.name,
                "args": {"urls": links, "page": page},
                "id": f"round-trip-{page}",
                "type": "tool_call",
            }
        )
        pages.append(message.artifact)
        page = message.artifact.next_page
    return pages


def test_every_citation_resolves_to_a_document(
    capsys: pytest.CaptureFixture[str],
) -> None:
    retriever = KapaRetriever()
    tool = KapaGetDocumentsTool()

    citations: dict[str, str] = {}
    for query in load_queries():
        documents = retriever.invoke(query)
        assert documents, f"No results for {query!r}; use queries the project answers."
        for document in documents:
            citations.setdefault(document.metadata["source"], query)
    links = [link for link in citations if link]

    pages = resolve(tool, links)
    results = [result for page in pages for result in page.results]
    documents_by_id = {
        document.document_id: document for page in pages for document in page.documents
    }

    assert [result.requested_url for result in results] == links
    missing = [
        f"{result.requested_url} (from {citations[result.requested_url or '']!r})"
        for result in results
        if result.status == "not_found"
    ]
    assert not missing, "Citations that resolve to nothing:\n" + "\n".join(missing)

    kinds = Counter(anchor_kind(result) for result in results)
    unavailable = [
        result.requested_url
        for result in results
        if result.document_id is not None
        and not documents_by_id[result.document_id].content_available
    ]
    with capsys.disabled():
        print("\nRound trip report")
        print(f"  citations resolved: {len(results)}")
        print(f"  citations without a link: {len(citations) - len(links)}")
        for kind in ("heading", "line", "page", "exact fragment-bearing link"):
            print(f"  {kind}: {kinds.get(kind, 0)}")
        print(f"  no anchor: {kinds.get('no anchor', 0)}")
        print(f"  found with unavailable content: {len(unavailable)}")
        for link in unavailable:
            print(f"    {link}")
