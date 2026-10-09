from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from importlib.metadata import version
from typing import Any
from urllib.parse import quote

import httpx
from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from langchain_kapa_ai.exceptions import (
    KapaAPIError,
    KapaAuthenticationError,
    KapaConnectionError,
    KapaNotFoundError,
    KapaRateLimitError,
    KapaResponseError,
    KapaServiceError,
    KapaValidationError,
)

DEFAULT_BASE_URL = "https://api.kapa.ai"
DEFAULT_TIMEOUT = 60.0
_MAX_DETAIL_CHARS = 500
_USER_AGENT = f"langchain-kapa-ai/{version('langchain-kapa-ai')}"


class KapaSettings(BaseModel):
    """Connection settings shared by the Kapa retriever and tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: SecretStr = Field(
        default_factory=secret_from_env(
            "KAPA_API_KEY",
            error_message=(
                "Pass `api_key` or set the KAPA_API_KEY environment variable."
            ),
        ),
        description="Project API key; read from KAPA_API_KEY when not passed.",
    )
    project_id: str = Field(
        default_factory=from_env(
            "KAPA_PROJECT_ID",
            error_message=(
                "Pass `project_id` or set the KAPA_PROJECT_ID environment variable."
            ),
        ),
        min_length=1,
        description="Kapa project to query; read from KAPA_PROJECT_ID when not passed.",
    )
    base_url: str = DEFAULT_BASE_URL
    """Base URL of the Kapa API."""
    timeout: float = Field(
        default=DEFAULT_TIMEOUT,
        gt=0,
        description="Timeout in seconds for each request.",
    )
    http_client: httpx.Client | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Client for synchronous requests; one per call when unset.",
    )
    http_async_client: httpx.AsyncClient | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Client for asynchronous requests; one per call when unset.",
    )

    def _kapa_client(self) -> KapaClient:
        return KapaClient(self)


class KapaClient:
    def __init__(self, settings: KapaSettings) -> None:
        self._settings = settings

    @property
    def timeout(self) -> float:
        return self._settings.timeout

    def url(self, endpoint: str) -> str:
        base_url = self._settings.base_url.rstrip("/")
        project_id = quote(self._settings.project_id, safe="")
        return f"{base_url}/query/v1/projects/{project_id}/{endpoint}/"

    def headers(self) -> dict[str, str]:
        return {
            "X-API-KEY": self._settings.api_key.get_secret_value(),
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        }

    @contextmanager
    def session(self) -> Iterator[KapaSession]:
        if self._settings.http_client is not None:
            yield KapaSession(self, self._settings.http_client)
            return
        with httpx.Client() as client:
            yield KapaSession(self, client)

    @asynccontextmanager
    async def async_session(self) -> AsyncIterator[KapaAsyncSession]:
        if self._settings.http_async_client is not None:
            yield KapaAsyncSession(self, self._settings.http_async_client)
            return
        async with httpx.AsyncClient() as client:
            yield KapaAsyncSession(self, client)

    def post(self, endpoint: str, body: dict[str, Any]) -> Any:
        with self.session() as session:
            return session.post(endpoint, body)

    async def apost(self, endpoint: str, body: dict[str, Any]) -> Any:
        async with self.async_session() as session:
            return await session.post(endpoint, body)


class KapaSession:
    def __init__(self, client: KapaClient, http: httpx.Client) -> None:
        self._client = client
        self._http = http

    def post(self, endpoint: str, body: dict[str, Any]) -> Any:
        try:
            response = self._http.post(
                self._client.url(endpoint),
                json=body,
                headers=self._client.headers(),
                timeout=self._client.timeout,
            )
        except httpx.TimeoutException as exc:
            raise _timeout_error(self._client.timeout) from exc
        except httpx.TransportError as exc:
            raise _transport_error(exc) from exc
        return _parse(response)


class KapaAsyncSession:
    def __init__(self, client: KapaClient, http: httpx.AsyncClient) -> None:
        self._client = client
        self._http = http

    async def post(self, endpoint: str, body: dict[str, Any]) -> Any:
        try:
            response = await self._http.post(
                self._client.url(endpoint),
                json=body,
                headers=self._client.headers(),
                timeout=self._client.timeout,
            )
        except httpx.TimeoutException as exc:
            raise _timeout_error(self._client.timeout) from exc
        except httpx.TransportError as exc:
            raise _transport_error(exc) from exc
        return _parse(response)


def _timeout_error(timeout: float) -> KapaConnectionError:
    return KapaConnectionError(f"The request to Kapa timed out after {timeout:g}s.")


def _transport_error(exc: httpx.TransportError) -> KapaConnectionError:
    return KapaConnectionError(f"Could not reach Kapa: {type(exc).__name__}.")


def _parse(response: httpx.Response) -> Any:
    if not response.is_success:
        raise _status_error(response)
    try:
        return response.json()
    except ValueError as exc:
        raise KapaResponseError("Kapa returned a response that is not JSON.") from exc


def _detail(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text[:_MAX_DETAIL_CHARS] or None


def _describe(detail: Any) -> str:
    if isinstance(detail, dict) and isinstance(detail.get("detail"), str):
        text = detail["detail"]
    elif detail is None:
        return ""
    elif isinstance(detail, str):
        text = detail
    else:
        text = json.dumps(detail, ensure_ascii=False)
    if len(text) > _MAX_DETAIL_CHARS:
        text = text[:_MAX_DETAIL_CHARS] + "..."
    return f": {text}"


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _status_error(response: httpx.Response) -> KapaAPIError:
    status = response.status_code
    detail = _detail(response)
    message = f"Kapa returned HTTP {status}{_describe(detail)}"
    if status in (401, 403):
        return KapaAuthenticationError(message, status_code=status, detail=detail)
    if status == 404:
        return KapaNotFoundError(message, status_code=status, detail=detail)
    if status == 429:
        return KapaRateLimitError(
            message,
            status_code=status,
            detail=detail,
            retry_after=_retry_after(response),
        )
    if 400 <= status < 500:
        return KapaValidationError(message, status_code=status, detail=detail)
    return KapaServiceError(message, status_code=status, detail=detail)
