from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal
from uuid import UUID

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from langchain_kapa_ai._client import KapaAsyncSession, KapaSession, KapaSettings
from langchain_kapa_ai.exceptions import KapaResponseError

_DOCUMENTS_PER_REQUEST = 5

_DESCRIPTION = (
    "Fetch whole documents from the Kapa knowledge base by source link or document "
    "ID. Use it when a search result is not enough and you need the complete "
    "document. Pass source links exactly as they appear in search results, "
    "including any part after '#'; Kapa matches each link exactly first, then "
    "without its fragment. For each requested link or ID, the result says "
    "whether a document was found and which stored link matched; the documents "
    "follow. Content can be truncated, and it is null when the document text is "
    "unavailable. When has_more is true, call again with next_page."
)


class KapaGetDocumentsInput(BaseModel):
    """Arguments of the Kapa document tool."""

    urls: list[Annotated[str, StringConstraints(min_length=1)]] | None = Field(
        default=None,
        description=(
            "Source links of the documents to fetch, exactly as they appear in "
            "search results. Kapa matches each link exactly first, then without "
            "its fragment."
        ),
    )
    document_ids: list[UUID] | None = Field(
        default=None, description="Kapa document IDs of the documents to fetch."
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Page of the requested documents to return, starting at 1.",
    )

    @model_validator(mode="after")
    def _require_a_request(self) -> KapaGetDocumentsInput:
        if not self.urls and not self.document_ids:
            msg = "Provide at least one of `urls` or `document_ids`."
            raise ValueError(msg)
        return self


class KapaDocument(BaseModel):
    """A document returned by the Kapa document tool."""

    document_id: str
    """Kapa document ID."""
    source_url: str | None
    """Stored link of the document; None when it has no link."""
    title: str
    """Title of the document."""
    content: str | None
    """Markdown content, possibly truncated; None when the text is unavailable."""
    content_available: bool
    """Whether the document text is available."""
    total_chars: int
    """Length of the full document before truncation."""
    truncated: bool
    """Whether the content was shortened to the per-document limit."""


class KapaDocumentRequestResult(BaseModel):
    """The outcome of one requested link or document ID."""

    requested_url: str | None = None
    """Link as requested, when the item is a link."""
    requested_document_id: str | None = None
    """Document ID as requested, when the item is an ID."""
    status: Literal["found", "not_found"]
    """Whether a document was found."""
    matched_url: str | None = None
    """Stored link of the found document, for link requests; it differs from the
    requested link when Kapa matched the link without its fragment."""
    document_id: str | None = None
    """ID of the document found."""


class KapaDocumentsPage(BaseModel):
    """One page of a Kapa document lookup."""

    page: int
    """The returned page, starting at 1."""
    page_size: int
    """Maximum number of requested links and document IDs per page."""
    total_requested: int
    """Number of distinct requested links and document IDs."""
    has_more: bool
    """Whether another page of requested links or document IDs exists."""
    next_page: int | None
    """Number of the next page; None on the last page."""
    results: list[KapaDocumentRequestResult]
    """Outcome of each requested link or ID on this page, in request order."""
    documents: list[KapaDocument]
    """Each document found on this page, once, in request order."""


class _DocumentResult(BaseModel):
    requested_url: str | None
    document_id: UUID
    source_url: str | None
    title: str
    content: str | None
    total_chars: int
    truncated: bool


class _DocumentsResponse(BaseModel):
    results: list[_DocumentResult]


@dataclass
class _PagePlan:
    page: int
    page_size: int
    total_requested: int
    urls: list[str]
    document_ids: list[UUID]

    @property
    def has_more(self) -> bool:
        return self.page * self.page_size < self.total_requested


def _plan_page(
    urls: list[str] | None,
    document_ids: list[UUID] | None,
    page: int,
    page_size: int,
) -> _PagePlan:
    items: list[str | UUID] = [
        *dict.fromkeys(urls or []),
        *dict.fromkeys(document_ids or []),
    ]
    page_items = items[(page - 1) * page_size : page * page_size]
    return _PagePlan(
        page=page,
        page_size=page_size,
        total_requested=len(items),
        urls=[item for item in page_items if isinstance(item, str)],
        document_ids=[item for item in page_items if isinstance(item, UUID)],
    )


def _batches(values: list[Any]) -> list[list[Any]]:
    return [
        values[start : start + _DOCUMENTS_PER_REQUEST]
        for start in range(0, len(values), _DOCUMENTS_PER_REQUEST)
    ]


def _parse_results(data: Any) -> list[_DocumentResult]:
    try:
        return _DocumentsResponse.model_validate(data).results
    except ValidationError as exc:
        msg = "Kapa returned a documents response with unexpected fields."
        raise KapaResponseError(msg) from exc


def _index_by_url(
    requested: list[str], results: list[_DocumentResult]
) -> dict[str, _DocumentResult]:
    wanted = set(requested)
    found: dict[str, _DocumentResult] = {}
    for result in results:
        if result.requested_url is None or result.requested_url not in wanted:
            msg = "Kapa returned a document for a link that was not requested."
            raise KapaResponseError(msg)
        found[result.requested_url] = result
    return found


def _index_by_id(
    requested: list[UUID], results: list[_DocumentResult]
) -> dict[UUID, _DocumentResult]:
    wanted = set(requested)
    found: dict[UUID, _DocumentResult] = {}
    for result in results:
        if result.document_id not in wanted:
            msg = "Kapa returned a document for an ID that was not requested."
            raise KapaResponseError(msg)
        found[result.document_id] = result
    return found


def _to_document(result: _DocumentResult) -> KapaDocument:
    return KapaDocument(
        document_id=str(result.document_id),
        source_url=result.source_url,
        title=result.title,
        content=result.content,
        content_available=result.content is not None,
        total_chars=result.total_chars,
        truncated=result.truncated,
    )


def _assemble(
    plan: _PagePlan,
    by_url: dict[str, _DocumentResult],
    by_id: dict[UUID, _DocumentResult],
) -> KapaDocumentsPage:
    results: list[KapaDocumentRequestResult] = []
    documents: dict[UUID, KapaDocument] = {}

    def found(result: _DocumentResult) -> str:
        documents.setdefault(result.document_id, _to_document(result))
        return str(result.document_id)

    for link in plan.urls:
        result = by_url.get(link)
        if result is None:
            results.append(
                KapaDocumentRequestResult(requested_url=link, status="not_found")
            )
            continue
        results.append(
            KapaDocumentRequestResult(
                requested_url=link,
                status="found",
                matched_url=result.source_url,
                document_id=found(result),
            )
        )

    for document_id in plan.document_ids:
        id_result = by_id.get(document_id)
        if id_result is None:
            results.append(
                KapaDocumentRequestResult(
                    requested_document_id=str(document_id), status="not_found"
                )
            )
            continue
        results.append(
            KapaDocumentRequestResult(
                requested_document_id=str(document_id),
                status="found",
                document_id=found(id_result),
            )
        )

    return KapaDocumentsPage(
        page=plan.page,
        page_size=plan.page_size,
        total_requested=plan.total_requested,
        has_more=plan.has_more,
        next_page=plan.page + 1 if plan.has_more else None,
        results=results,
        documents=list(documents.values()),
    )


class KapaDocumentSettings(BaseModel):
    """Document lookup settings shared by the Kapa document tool and toolkit."""

    source_group_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Source groups to restrict the lookup to; all sources when unset.",
    )
    max_chars_per_document: int | None = Field(
        default=None,
        ge=1,
        description="Maximum characters per document; Kapa's default if unset.",
    )
    page_size: int = Field(
        default=5,
        ge=1,
        description="Number of requested links and document IDs per page.",
    )


class KapaGetDocumentsTool(KapaSettings, KapaDocumentSettings, BaseTool):
    """Fetch whole documents from a Kapa knowledge base by link or document ID.

    Kapa matches each link exactly first, then without its fragment, and each
    result reports the stored link that matched. A page covers `page_size`
    distinct requested links and document IDs, so it returns at most that many
    documents.
    """

    name: str = "kapa_get_documents"
    """Tool name shown to the model."""
    description: str = _DESCRIPTION
    """Tool description shown to the model."""
    args_schema: ArgsSchema | None = KapaGetDocumentsInput
    """Schema of the arguments the model passes."""
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    """Returns JSON text for the model and a `KapaDocumentsPage` artifact."""

    def _request_body(self, key: str, values: list[Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            key: [str(value) for value in values],
            "page": 1,
            "page_size": len(values),
        }
        if self.source_group_ids is not None:
            body["source_group_ids_include"] = list(self.source_group_ids)
        if self.max_chars_per_document is not None:
            body["max_chars_per_document"] = self.max_chars_per_document
        return body

    def _fetch_urls(
        self, session: KapaSession, urls: list[str]
    ) -> dict[str, _DocumentResult]:
        found: dict[str, _DocumentResult] = {}
        for batch in _batches(urls):
            data = session.post("documents", self._request_body("urls", batch))
            found.update(_index_by_url(batch, _parse_results(data)))
        return found

    def _fetch_ids(
        self, session: KapaSession, document_ids: list[UUID]
    ) -> dict[UUID, _DocumentResult]:
        found: dict[UUID, _DocumentResult] = {}
        for batch in _batches(document_ids):
            data = session.post("documents", self._request_body("document_ids", batch))
            found.update(_index_by_id(batch, _parse_results(data)))
        return found

    async def _afetch_urls(
        self, session: KapaAsyncSession, urls: list[str]
    ) -> dict[str, _DocumentResult]:
        found: dict[str, _DocumentResult] = {}
        for batch in _batches(urls):
            data = await session.post("documents", self._request_body("urls", batch))
            found.update(_index_by_url(batch, _parse_results(data)))
        return found

    async def _afetch_ids(
        self, session: KapaAsyncSession, document_ids: list[UUID]
    ) -> dict[UUID, _DocumentResult]:
        found: dict[UUID, _DocumentResult] = {}
        for batch in _batches(document_ids):
            body = self._request_body("document_ids", batch)
            data = await session.post("documents", body)
            found.update(_index_by_id(batch, _parse_results(data)))
        return found

    def _run(
        self,
        urls: list[str] | None = None,
        document_ids: list[UUID] | None = None,
        page: int = 1,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, KapaDocumentsPage]:
        plan = _plan_page(urls, document_ids, page, self.page_size)
        with self._kapa_client().session() as session:
            by_url = self._fetch_urls(session, plan.urls)
            by_id = self._fetch_ids(session, plan.document_ids)
        result = _assemble(plan, by_url, by_id)
        return result.model_dump_json(), result

    async def _arun(
        self,
        urls: list[str] | None = None,
        document_ids: list[UUID] | None = None,
        page: int = 1,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, KapaDocumentsPage]:
        plan = _plan_page(urls, document_ids, page, self.page_size)
        async with self._kapa_client().async_session() as session:
            by_url = await self._afetch_urls(session, plan.urls)
            by_id = await self._afetch_ids(session, plan.document_ids)
        result = _assemble(plan, by_url, by_id)
        return result.model_dump_json(), result
