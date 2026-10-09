from __future__ import annotations

from typing import Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field, ValidationError

from langchain_kapa_ai._client import KapaSettings
from langchain_kapa_ai.exceptions import KapaResponseError

RetrievalMode = Literal["default", "deep"]

_PER_CALL_SETTINGS = frozenset({"mode", "top_k", "max_chars"})


class KapaEndUser(BaseModel):
    """End user attributed to retrieval queries in Kapa analytics."""

    email: str | None = None
    """Email address of the end user."""
    unique_client_id: str | None = None
    """Identifier of the end user in your application."""
    metadata: dict[str, Any] | None = None
    """Additional attributes of the end user."""


class _SearchSettings(BaseModel):
    mode: RetrievalMode
    top_k: int | None = Field(default=None, ge=1)
    max_chars: int | None = Field(default=None, ge=1)


class _RetrievedChunk(BaseModel):
    source_url: str
    content: str


class KapaRetrievalSettings(BaseModel):
    """Search settings shared by the Kapa retriever and toolkit."""

    mode: RetrievalMode = "default"
    """Retrieval mode: the faster `default` or the more thorough `deep`."""
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of chunks; Kapa's default applies when unset.",
    )
    max_chars: int | None = Field(
        default=None,
        ge=1,
        description="Maximum characters across all chunks; Kapa's default if unset.",
    )
    source_group_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Source groups to restrict retrieval to; all sources when unset.",
    )
    integration_id: str | None = None
    """Integration that analytics attributes queries to."""
    redact_query: bool = False
    """Whether to redact the query in analytics."""
    end_user: KapaEndUser | None = None
    """End user that analytics attributes queries to."""


class KapaRetriever(KapaSettings, KapaRetrievalSettings, BaseRetriever):
    """Retrieve relevant chunks from a Kapa knowledge base.

    Each document's `page_content` is the chunk text and `metadata["source"]`
    is its citation link, unchanged from Kapa.
    """

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        body = self._request_body(query, kwargs)
        return _to_documents(self._kapa_client().post("retrieval", body))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        body = self._request_body(query, kwargs)
        return _to_documents(await self._kapa_client().apost("retrieval", body))

    def _request_body(self, query: str, overrides: dict[str, Any]) -> dict[str, Any]:
        unsupported = sorted(set(overrides) - _PER_CALL_SETTINGS)
        if unsupported:
            msg = f"Unsupported retrieval arguments: {', '.join(unsupported)}."
            raise TypeError(msg)
        search = _SearchSettings.model_validate(
            {
                "mode": self.mode,
                "top_k": self.top_k,
                "max_chars": self.max_chars,
                **overrides,
            }
        )
        body: dict[str, Any] = {"query": query, "mode": search.mode}
        if search.top_k is not None:
            body["top_k"] = search.top_k
        if search.max_chars is not None:
            body["max_chars"] = search.max_chars
        if self.source_group_ids is not None:
            body["source_group_ids_include"] = list(self.source_group_ids)
        if self.integration_id is not None:
            body["integration_id"] = self.integration_id
        if self.redact_query:
            body["redact_query"] = True
        if self.end_user is not None:
            body["user"] = self.end_user.model_dump(exclude_none=True)
        return body


def _to_documents(data: Any) -> list[Document]:
    if not isinstance(data, list):
        msg = "Kapa returned a retrieval response that is not a list."
        raise KapaResponseError(msg)
    try:
        chunks = [_RetrievedChunk.model_validate(item) for item in data]
    except ValidationError as exc:
        msg = "Kapa returned a retrieval result with unexpected fields."
        raise KapaResponseError(msg) from exc
    return [
        Document(page_content=chunk.content, metadata={"source": chunk.source_url})
        for chunk in chunks
    ]
