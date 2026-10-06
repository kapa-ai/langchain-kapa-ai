from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_kapa_ai import KapaGetDocumentsTool


class TestKapaGetDocumentsToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return KapaGetDocumentsTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": "test-key", "project_id": "test-project"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"urls": ["https://docs.example.com/guide#install"]}

    @property
    def init_from_env_params(
        self,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return (
            {"KAPA_API_KEY": "env-key", "KAPA_PROJECT_ID": "env-project"},
            {},
            {"api_key": "env-key", "project_id": "env-project"},
        )
