from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from langchain_kapa_ai import (
    KapaDocumentsPage,
    KapaGetDocumentsTool,
    KapaResponseError,
    KapaServiceError,
)
from tests.unit_tests.fake_kapa import FakeKapa, StoredDocument

GUIDE = StoredDocument("11111111-1111-1111-1111-111111111111", "https://d.test/guide")
CODE = StoredDocument(
    "22222222-2222-2222-2222-222222222222", "https://d.test/client.py"
)
PDF = StoredDocument(
    "33333333-3333-3333-3333-333333333333", "https://d.test/manual.pdf", content=None
)
UPLOAD = StoredDocument("44444444-4444-4444-4444-444444444444", "https://d.test/hb#v2")
UPLOAD_V3 = StoredDocument(
    "88888888-8888-8888-8888-888888888888", "https://d.test/hb#v3"
)
EMPTY = StoredDocument(
    "55555555-5555-5555-5555-555555555555", "https://d.test/empty", content=""
)
LONG = StoredDocument(
    "66666666-6666-6666-6666-666666666666",
    "https://d.test/long",
    content="abc",
    total_chars=10_000,
    truncated=True,
)
ALL = [GUIDE, CODE, PDF, UPLOAD, UPLOAD_V3, EMPTY, LONG]


def make_tool(kapa: FakeKapa, **settings: Any) -> KapaGetDocumentsTool:
    kapa.documents = ALL
    return KapaGetDocumentsTool(**kapa.settings(), **settings)


def lookup(tool: KapaGetDocumentsTool, **args: Any) -> KapaDocumentsPage:
    message = tool.invoke(
        {"name": tool.name, "args": args, "id": "call-1", "type": "tool_call"}
    )
    assert isinstance(message, ToolMessage)
    assert isinstance(message.artifact, KapaDocumentsPage)
    assert json.loads(str(message.content)) == message.artifact.model_dump(mode="json")
    return message.artifact


def document_ids(page: KapaDocumentsPage) -> list[str]:
    return [document.document_id for document in page.documents]


def requested_urls(kapa: FakeKapa) -> list[list[str]]:
    return [body["urls"] for body in kapa.bodies if "urls" in body]


def test_documents_are_returned_in_request_order(kapa: FakeKapa) -> None:
    urls = [GUIDE.source_url, "https://d.test/client.py#L10-L24", UPLOAD.source_url]

    page = lookup(make_tool(kapa), urls=urls)

    assert document_ids(page) == [
        GUIDE.document_id,
        CODE.document_id,
        UPLOAD.document_id,
    ]
    assert page.total_requested == 3
    assert requested_urls(kapa) == [urls]


def test_link_the_backend_omits_yields_no_document(kapa: FakeKapa) -> None:
    urls = ["https://d.test/missing", "https://d.test/missing#a"]

    page = lookup(make_tool(kapa), urls=urls)

    assert page.documents == []
    assert page.total_requested == 2
    assert requested_urls(kapa) == [urls]


def test_links_are_stripped_before_sending(kapa: FakeKapa) -> None:
    page = lookup(make_tool(kapa), urls=[f"  {GUIDE.source_url}\n", GUIDE.source_url])

    assert document_ids(page) == [GUIDE.document_id]
    assert page.total_requested == 1
    assert requested_urls(kapa) == [[GUIDE.source_url]]


def test_duplicate_links_are_requested_once(kapa: FakeKapa) -> None:
    page = lookup(make_tool(kapa), urls=[GUIDE.source_url, GUIDE.source_url])

    assert document_ids(page) == [GUIDE.document_id]
    assert page.total_requested == 1
    assert requested_urls(kapa) == [[GUIDE.source_url]]


def test_links_to_one_document_list_it_once(kapa: FakeKapa) -> None:
    citations = ["https://d.test/guide#a", "https://d.test/guide#b"]

    page = lookup(make_tool(kapa), urls=citations)

    assert document_ids(page) == [GUIDE.document_id]
    assert page.total_requested == 2
    assert requested_urls(kapa) == [citations]


def test_repeated_document_keeps_its_first_position(kapa: FakeKapa) -> None:
    urls = ["https://d.test/guide#a", CODE.source_url, "https://d.test/guide#b"]

    page = lookup(make_tool(kapa), urls=urls)

    assert document_ids(page) == [GUIDE.document_id, CODE.document_id]
    assert requested_urls(kapa) == [urls]


def test_document_ids_are_deduplicated_case_insensitively(kapa: FakeKapa) -> None:
    upper = CODE.document_id.upper()
    missing = "77777777-7777-7777-7777-777777777777"

    page = lookup(make_tool(kapa), document_ids=[CODE.document_id, upper, missing])

    assert document_ids(page) == [CODE.document_id]
    assert page.total_requested == 2
    assert [body["document_ids"] for body in kapa.bodies] == [
        [CODE.document_id, missing]
    ]


def test_urls_and_ids_are_requested_separately(kapa: FakeKapa) -> None:
    page = lookup(
        make_tool(kapa), urls=[GUIDE.source_url], document_ids=[GUIDE.document_id]
    )

    assert document_ids(page) == [GUIDE.document_id]
    assert page.total_requested == 2
    assert [sorted(body) for body in kapa.bodies] == [
        ["page", "page_size", "urls"],
        ["document_ids", "page", "page_size"],
    ]


def test_documents_found_by_link_precede_those_found_by_id(kapa: FakeKapa) -> None:
    page = lookup(
        make_tool(kapa), urls=[GUIDE.source_url], document_ids=[CODE.document_id]
    )

    assert document_ids(page) == [GUIDE.document_id, CODE.document_id]


def test_backend_requests_stay_within_page_limit(kapa: FakeKapa) -> None:
    urls = [f"https://d.test/guide#s{i}" for i in range(7)]

    page = lookup(make_tool(kapa, page_size=7), urls=urls)

    assert document_ids(page) == [GUIDE.document_id]
    assert requested_urls(kapa) == [urls[:5], urls[5:]]
    assert all(body["page"] == 1 for body in kapa.bodies)
    assert all(body["page_size"] == len(body["urls"]) for body in kapa.bodies)


def test_source_groups_and_content_bound_apply_to_every_request(
    kapa: FakeKapa,
) -> None:
    tool = make_tool(kapa, source_group_ids=["group-a"], max_chars_per_document=100)

    lookup(tool, urls=["https://d.test/guide#a"], document_ids=[CODE.document_id])

    assert len(kapa.bodies) == 2
    for body in kapa.bodies:
        assert body["source_group_ids_include"] == ["group-a"]
        assert body["max_chars_per_document"] == 100


def test_pages_follow_requested_items_not_matches(kapa: FakeKapa) -> None:
    missing = [f"https://d.test/missing-{i}" for i in range(2)]
    urls = [*missing, GUIDE.source_url, CODE.source_url]
    tool = make_tool(kapa, page_size=2)

    first = lookup(tool, urls=urls)
    second = lookup(tool, urls=urls, page=2)
    third = lookup(tool, urls=urls, page=3)

    assert first.documents == []
    assert (first.total_requested, first.has_more, first.next_page) == (4, True, 2)
    assert document_ids(second) == [GUIDE.document_id, CODE.document_id]
    assert (second.has_more, second.next_page) == (False, None)
    assert (third.documents, third.has_more) == ([], False)
    assert requested_urls(kapa) == [missing, urls[2:]]


def test_each_distinct_link_is_one_requested_item(kapa: FakeKapa) -> None:
    urls = ["https://d.test/guide#a", "https://d.test/guide#b", CODE.source_url]

    page = lookup(make_tool(kapa, page_size=1), urls=urls)

    assert page.total_requested == 3
    assert document_ids(page) == [GUIDE.document_id]
    assert (page.has_more, page.next_page) == (True, 2)
    assert requested_urls(kapa) == [urls[:1]]


def test_page_returns_at_most_page_size_documents(kapa: FakeKapa) -> None:
    urls = [UPLOAD.source_url, UPLOAD_V3.source_url]
    tool = make_tool(kapa, page_size=1)

    first = lookup(tool, urls=urls)
    second = lookup(tool, urls=urls, page=2)

    assert document_ids(first) == [UPLOAD.document_id]
    assert (first.total_requested, first.has_more) == (2, True)
    assert document_ids(second) == [UPLOAD_V3.document_id]
    assert not second.has_more


def test_document_fields_distinguish_unavailable_empty_and_truncated(
    kapa: FakeKapa,
) -> None:
    page = lookup(
        make_tool(kapa), urls=[PDF.source_url, EMPTY.source_url, LONG.source_url]
    )

    pdf, empty, long = page.documents
    assert (pdf.content, pdf.content_available, pdf.total_chars) == (None, False, 0)
    assert (empty.content, empty.content_available) == ("", True)
    assert (long.truncated, long.total_chars, long.content) == (True, 10_000, "abc")
    content = json.loads(page.model_dump_json())
    assert content["documents"][0]["content"] is None
    assert content["documents"][1]["content"] == ""


async def test_async_matches_sync(kapa: FakeKapa) -> None:
    tool = make_tool(kapa, page_size=2)
    args = {
        "urls": ["https://d.test/guide#a", UPLOAD.source_url, "https://d.test/x#y"],
        "document_ids": [PDF.document_id],
        "page": 1,
    }

    sync_page = lookup(tool, **args)
    async_message = await tool.ainvoke(
        {"name": tool.name, "args": args, "id": "call-2", "type": "tool_call"}
    )

    assert async_message.artifact == sync_page
    half = len(kapa.bodies) // 2
    assert kapa.bodies[:half] == kapa.bodies[half:]


def test_plain_invoke_returns_content_text(kapa: FakeKapa) -> None:
    content = make_tool(kapa).invoke({"urls": [GUIDE.source_url]})

    documents = json.loads(content)["documents"]
    assert [document["document_id"] for document in documents] == [GUIDE.document_id]


def test_request_without_links_or_ids_is_rejected(kapa: FakeKapa) -> None:
    with pytest.raises(ValidationError):
        make_tool(kapa).invoke({"urls": [], "document_ids": []})

    assert kapa.requests == []


def test_invalid_document_id_is_rejected(kapa: FakeKapa) -> None:
    with pytest.raises(ValidationError):
        make_tool(kapa).invoke({"document_ids": ["not-a-uuid"]})


def test_failed_request_does_not_become_an_empty_result(kapa: FakeKapa) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    kapa.override = handler

    with pytest.raises(KapaServiceError):
        make_tool(kapa).invoke({"urls": [GUIDE.source_url]})
    assert len(kapa.requests) == 1


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"results": [{"document_id": GUIDE.document_id}]},
        {
            "results": [
                {
                    key: value
                    for key, value in GUIDE.as_result().items()
                    if key != "title"
                }
            ]
        },
    ],
)
def test_malformed_responses_raise(kapa: FakeKapa, payload: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    kapa.override = handler

    with pytest.raises(KapaResponseError, match="unexpected fields"):
        make_tool(kapa).invoke({"urls": [GUIDE.source_url]})
