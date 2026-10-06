from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

API_KEY = "test-secret-key"
PROJECT_ID = "test-project"


@dataclass
class StoredDocument:
    document_id: str
    source_url: str | None
    title: str = "Title"
    content: str | None = "Content"
    total_chars: int | None = None
    truncated: bool = False

    def as_result(self) -> dict[str, Any]:
        total = self.total_chars
        if total is None:
            total = len(self.content) if self.content is not None else 0
        return {
            "document_id": self.document_id,
            "source_url": self.source_url,
            "title": self.title,
            "content": self.content,
            "total_chars": total,
            "truncated": self.truncated,
        }


@dataclass
class FakeKapa:
    chunks: list[dict[str, Any]] = field(default_factory=list)
    documents: list[StoredDocument] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)
    override: Callable[[httpx.Request], httpx.Response] | None = None

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.override is not None:
            return self.override(request)
        if request.url.path.endswith("/retrieval/"):
            return httpx.Response(200, json=self.chunks)
        if request.url.path.endswith("/documents/"):
            return httpx.Response(
                200, json=self._documents(json.loads(request.content))
            )
        return httpx.Response(404, json={"detail": "Not found."})

    def _documents(self, body: dict[str, Any]) -> dict[str, Any]:
        by_url = {doc.source_url: doc for doc in self.documents if doc.source_url}
        by_id = {doc.document_id.lower(): doc for doc in self.documents}
        urls = list(dict.fromkeys(body.get("urls") or []))
        ids = list(dict.fromkeys(i.lower() for i in body.get("document_ids") or []))
        requests = [by_url.get(url) for url in urls] + [by_id.get(i) for i in ids]
        page, page_size = body.get("page", 1), body.get("page_size", 5)
        window = requests[(page - 1) * page_size : page * page_size]
        results = [doc.as_result() for doc in window if doc is not None]
        return {
            "results": results,
            "page": page,
            "page_size": page_size,
            "total_items": sum(doc is not None for doc in requests),
        }

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def settings(self) -> dict[str, Any]:
        return {
            "api_key": API_KEY,
            "project_id": PROJECT_ID,
            "http_client": self.client(),
            "http_async_client": self.async_client(),
        }
