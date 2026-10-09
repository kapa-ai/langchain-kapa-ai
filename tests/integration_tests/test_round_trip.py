from __future__ import annotations

import re
from collections import Counter

import pytest

from langchain_kapa_ai import (
    KapaDocument,
    KapaDocumentsPage,
    KapaGetDocumentsTool,
    KapaRetriever,
)
from tests.integration_tests.live import load_queries, requires_queries

pytestmark = requires_queries


def anchor_kind(link: str) -> str:
    if "#" not in link:
        return "no anchor"
    fragment = link.partition("#")[2]
    if re.fullmatch(r"L\d+(-L\d+)?", fragment):
        return "line"
    if re.fullmatch(r"page=\d+", fragment):
        return "page"
    return "heading"


def resolves(link: str, source_urls: set[str]) -> bool:
    return link in source_urls or link.partition("#")[0] in source_urls


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
        chunks = retriever.invoke(query)
        assert chunks, f"No results for {query!r}; use queries the project answers."
        for chunk in chunks:
            citations.setdefault(chunk.metadata["source"], query)
    links = [link for link in citations if link]

    pages = resolve(tool, links)
    documents: dict[str, KapaDocument] = {
        document.document_id: document for page in pages for document in page.documents
    }
    source_urls = {
        document.source_url for document in documents.values() if document.source_url
    }

    missing = [
        f"{link} (from {citations[link]!r})"
        for link in links
        if not resolves(link, source_urls)
    ]
    assert not missing, "Citations that resolve to nothing:\n" + "\n".join(missing)

    kinds = Counter(anchor_kind(link) for link in links)
    unavailable = [
        document.source_url or document.document_id
        for document in documents.values()
        if not document.content_available
    ]
    with capsys.disabled():
        print("\nRound trip report")
        print(f"  citations resolved: {len(links)}")
        print(f"  citations without a link: {len(citations) - len(links)}")
        print(f"  documents found: {len(documents)}")
        for kind in ("heading", "line", "page", "no anchor"):
            print(f"  {kind}: {kinds.get(kind, 0)}")
        print(f"  found with unavailable content: {len(unavailable)}")
        for link in unavailable:
            print(f"    {link}")
