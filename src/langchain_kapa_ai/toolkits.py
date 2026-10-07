from __future__ import annotations

from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, BaseToolkit, create_retriever_tool

from langchain_kapa_ai._client import KapaSettings
from langchain_kapa_ai.retrievers import KapaRetrievalSettings, KapaRetriever
from langchain_kapa_ai.tools import KapaDocumentSettings, KapaGetDocumentsTool

SEARCH_TOOL_NAME = "kapa_search_knowledge_base"
SEARCH_TOOL_DESCRIPTION = (
    "Search the project's knowledge base. Returns relevant passages, each with its "
    "source link."
)
SEARCH_DOCUMENT_PROMPT = "Source: {source}\n{page_content}"


class KapaToolkit(
    KapaSettings, KapaRetrievalSettings, KapaDocumentSettings, BaseToolkit
):
    """The Kapa search tool and document tool, configured from one set of settings."""

    def _settings(self, *models: type[Any]) -> dict[str, Any]:
        names = set(KapaSettings.model_fields).union(
            *(model.model_fields for model in models)
        )
        return {name: getattr(self, name) for name in names}

    def get_tools(self) -> list[BaseTool]:
        search = create_retriever_tool(
            KapaRetriever(**self._settings(KapaRetrievalSettings)),
            name=SEARCH_TOOL_NAME,
            description=SEARCH_TOOL_DESCRIPTION,
            document_prompt=PromptTemplate.from_template(SEARCH_DOCUMENT_PROMPT),
            response_format="content_and_artifact",
        )
        documents = KapaGetDocumentsTool(**self._settings(KapaDocumentSettings))
        return [search, documents]
