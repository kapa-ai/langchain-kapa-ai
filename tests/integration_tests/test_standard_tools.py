from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_kapa_ai import KapaGetDocumentsTool, KapaToolkit
from tests.integration_tests.live import requires_credentials

pytestmark = requires_credentials


class TestKapaGetDocumentsToolLive(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return KapaGetDocumentsTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"urls": ["https://docs.example.com/langchain-kapa-ai/not-indexed"]}


class TestKapaToolkitSearchToolLive(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return KapaToolkit().get_tools()[0]

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "How do I get started?"}
