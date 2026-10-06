from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain_core.documents import Document

from langchain_kapa_ai import KapaEndUser, KapaResponseError, KapaRetriever
from tests.unit_tests.fake_kapa import FakeKapa

CHUNKS = [
    {"source_url": "https://docs.example.com/guide#install", "content": "Install it."},
    {"source_url": "https://code.example.com/client.py#L10-L24", "content": "def a()"},
    {"source_url": "https://docs.example.com/manual.pdf#page=3", "content": "Page 3"},
    {"source_url": "https://docs.example.com/handbook#v2", "content": "Handbook"},
    {"source_url": "", "content": "No link"},
]


def documents(chunks: list[dict[str, Any]]) -> list[Document]:
    return [
        Document(
            page_content=chunk["content"], metadata={"source": chunk["source_url"]}
        )
        for chunk in chunks
    ]


def test_minimal_request_body(kapa: FakeKapa) -> None:
    KapaRetriever(**kapa.settings()).invoke("How do I install it?")

    assert kapa.bodies == [{"query": "How do I install it?", "mode": "default"}]


def test_full_request_body(kapa: FakeKapa) -> None:
    retriever = KapaRetriever(
        **kapa.settings(),
        mode="deep",
        top_k=4,
        max_chars=9000,
        source_group_ids=["group-a", "group-b"],
        integration_id="integration-1",
        redact_query=True,
        end_user=KapaEndUser(email="user@example.com", metadata={"plan": "pro"}),
    )

    retriever.invoke("question")

    assert kapa.bodies == [
        {
            "query": "question",
            "mode": "deep",
            "top_k": 4,
            "max_chars": 9000,
            "source_group_ids_include": ["group-a", "group-b"],
            "integration_id": "integration-1",
            "redact_query": True,
            "user": {"email": "user@example.com", "metadata": {"plan": "pro"}},
        }
    ]


def test_empty_source_group_list_is_sent(kapa: FakeKapa) -> None:
    KapaRetriever(**kapa.settings(), source_group_ids=[]).invoke("question")

    assert kapa.bodies[0]["source_group_ids_include"] == []


def test_per_call_settings_override_configuration(kapa: FakeKapa) -> None:
    retriever = KapaRetriever(**kapa.settings(), top_k=10)

    retriever.invoke("question", top_k=2, mode="deep", max_chars=500)
    retriever.invoke("question")

    assert kapa.bodies[0] == {
        "query": "question",
        "mode": "deep",
        "top_k": 2,
        "max_chars": 500,
    }
    assert kapa.bodies[1] == {"query": "question", "mode": "default", "top_k": 10}


def test_unknown_per_call_settings_are_rejected(kapa: FakeKapa) -> None:
    with pytest.raises(TypeError, match="source_group_ids"):
        KapaRetriever(**kapa.settings()).invoke("question", source_group_ids=["a"])

    assert kapa.requests == []


def test_invalid_per_call_settings_are_rejected(kapa: FakeKapa) -> None:
    with pytest.raises(ValueError, match="mode"):
        KapaRetriever(**kapa.settings()).invoke("question", mode="fast")

    assert kapa.requests == []


def test_results_keep_order_content_and_citation_links(kapa: FakeKapa) -> None:
    kapa.chunks = CHUNKS

    result = KapaRetriever(**kapa.settings()).invoke("question")

    assert result == documents(CHUNKS)
    assert all(document.id is None for document in result)


def test_fewer_results_than_requested_are_valid(kapa: FakeKapa) -> None:
    kapa.chunks = CHUNKS[:2]

    result = KapaRetriever(**kapa.settings(), top_k=10).invoke("question")

    assert result == documents(CHUNKS[:2])


def test_empty_result_is_valid(kapa: FakeKapa) -> None:
    assert KapaRetriever(**kapa.settings()).invoke("question") == []


async def test_async_matches_sync(kapa: FakeKapa) -> None:
    kapa.chunks = CHUNKS
    retriever = KapaRetriever(**kapa.settings(), mode="deep", top_k=3)

    sync_result = retriever.invoke("question", max_chars=100)
    async_result = await retriever.ainvoke("question", max_chars=100)

    assert async_result == sync_result
    assert kapa.bodies[0] == kapa.bodies[1]


def test_batch(kapa: FakeKapa) -> None:
    kapa.chunks = CHUNKS[:1]

    results = KapaRetriever(**kapa.settings()).batch(["one", "two"])

    assert results == [documents(CHUNKS[:1])] * 2
    assert sorted(body["query"] for body in kapa.bodies) == ["one", "two"]


async def test_abatch(kapa: FakeKapa) -> None:
    kapa.chunks = CHUNKS[:1]

    results = await KapaRetriever(**kapa.settings()).abatch(["one", "two"])

    assert results == [documents(CHUNKS[:1])] * 2


@pytest.mark.parametrize(
    "payload",
    [
        {"results": []},
        [{"source_url": "https://a.test"}],
        [{"source_url": None, "content": "text"}],
        [{"source_url": "https://a.test", "content": 3}],
    ],
)
def test_malformed_responses_raise(kapa: FakeKapa, payload: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    kapa.override = handler

    with pytest.raises(KapaResponseError):
        KapaRetriever(**kapa.settings()).invoke("question")
