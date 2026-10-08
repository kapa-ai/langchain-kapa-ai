from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_kapa_ai import (
    KapaDocumentsPage,
    KapaEndUser,
    KapaGetDocumentsTool,
    KapaToolkit,
)
from tests.unit_tests.fake_kapa import API_KEY, FakeKapa, StoredDocument

CHUNK = {"source_url": "https://d.test/guide#install", "content": "Install it."}
GUIDE = StoredDocument("11111111-1111-1111-1111-111111111111", "https://d.test/guide")


def call(tool: BaseTool, args: dict[str, Any]) -> ToolMessage:
    message = tool.invoke(
        {"name": tool.name, "args": args, "id": "1", "type": "tool_call"}
    )
    assert isinstance(message, ToolMessage)
    return message


def test_returns_search_and_document_tools_in_order(kapa: FakeKapa) -> None:
    tools = KapaToolkit(**kapa.settings()).get_tools()

    assert [tool.name for tool in tools] == [
        "kapa_search_knowledge_base",
        "kapa_get_documents",
    ]
    assert isinstance(tools[1], KapaGetDocumentsTool)
    assert tools[0].response_format == "content_and_artifact"


def test_search_output_shows_source_links(kapa: FakeKapa) -> None:
    kapa.chunks = [CHUNK]
    search, _ = KapaToolkit(**kapa.settings()).get_tools()

    message = call(search, {"query": "How do I install it?"})

    assert message.text == "Source: https://d.test/guide#install\nInstall it."
    assert message.artifact == [
        Document(page_content="Install it.", metadata={"source": CHUNK["source_url"]})
    ]


def test_settings_reach_both_tools(kapa: FakeKapa) -> None:
    kapa.chunks = [CHUNK]
    kapa.documents = [GUIDE]
    toolkit = KapaToolkit(
        **kapa.settings(),
        base_url="https://kapa.test",
        mode="deep",
        top_k=4,
        max_chars=900,
        source_group_ids=["group-a"],
        integration_id="integration-1",
        redact_query=True,
        end_user=KapaEndUser(unique_client_id="user-1"),
        max_chars_per_document=100,
        page_size=1,
    )
    search, documents = toolkit.get_tools()

    call(search, {"query": "question"})
    page = call(documents, {"urls": [GUIDE.source_url, "https://d.test/x"]}).artifact

    assert kapa.bodies[0] == {
        "query": "question",
        "mode": "deep",
        "top_k": 4,
        "max_chars": 900,
        "source_group_ids_include": ["group-a"],
        "integration_id": "integration-1",
        "redact_query": True,
        "user": {"unique_client_id": "user-1"},
    }
    assert kapa.bodies[1]["source_group_ids_include"] == ["group-a"]
    assert kapa.bodies[1]["max_chars_per_document"] == 100
    assert isinstance(page, KapaDocumentsPage)
    assert (page.page_size, page.has_more) == (1, True)
    assert all(str(r.url).startswith("https://kapa.test/") for r in kapa.requests)
    assert all(r.headers["X-API-KEY"] == API_KEY for r in kapa.requests)


async def test_async_tools_use_the_fake_transport(kapa: FakeKapa) -> None:
    kapa.chunks = [CHUNK]
    kapa.documents = [GUIDE]
    search, documents = KapaToolkit(**kapa.settings()).get_tools()

    found = await search.ainvoke({"query": "question"})
    looked_up = await documents.ainvoke({"urls": [CHUNK["source_url"]]})

    assert "Source: https://d.test/guide#install" in found
    assert f'"matched_url":"{GUIDE.source_url}"' in looked_up
    assert len(kapa.requests) == 2


def test_secrets_stay_out_of_the_toolkit_representation(kapa: FakeKapa) -> None:
    assert API_KEY not in repr(KapaToolkit(**kapa.settings()))


class TestKapaToolkitSearchToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return KapaToolkit(api_key="test-key", project_id="test-project").get_tools()[0]

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "How do I install it?"}
