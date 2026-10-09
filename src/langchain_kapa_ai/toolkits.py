from __future__ import annotations

from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, BaseToolkit, create_retriever_tool

from langchain_kapa_ai._client import KapaSettings
from langchain_kapa_ai.retrievers import KapaRetrievalSettings, KapaRetriever
from langchain_kapa_ai.tools import KapaDocumentSettings, KapaGetDocumentsTool

SEARCH_TOOL_NAME = "search_knowledge_sources"
SEARCH_TOOL_DESCRIPTION = (
    "Perform semantic retrieval over the knowledge sources and return the most "
    'relevant chunks for a given query. A "chunk" is a short, self-contained '
    "snippet of text taken from a single page or item within these sources (for "
    "example, part of a documentation page) and includes its source URL and "
    "markdown content. Chunks are returned in descending order of relevance to "
    "the query. If the knowledge sources do not contain information relevant to "
    "the query, the returned chunks may be only weakly related or entirely "
    "unrelated."
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
