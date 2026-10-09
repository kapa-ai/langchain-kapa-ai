from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from langchain_kapa_ai import (
    KapaAuthenticationError,
    KapaConnectionError,
    KapaError,
    KapaGetDocumentsTool,
    KapaNotFoundError,
    KapaRateLimitError,
    KapaResponseError,
    KapaRetriever,
    KapaServiceError,
    KapaValidationError,
)
from tests.unit_tests.fake_kapa import API_KEY, PROJECT_ID, FakeKapa


def respond(response: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return response

    return handler


def test_request_targets_project_endpoint_with_key(kapa: FakeKapa) -> None:
    KapaRetriever(**kapa.settings()).invoke("question")

    request = kapa.requests[0]
    assert request.method == "POST"
    assert str(request.url) == (
        f"https://api.kapa.ai/query/v1/projects/{PROJECT_ID}/retrieval/"
    )
    assert request.headers["X-API-KEY"] == API_KEY
    assert request.headers["User-Agent"].startswith("langchain-kapa-ai/")


def test_base_url_and_project_are_joined_safely(kapa: FakeKapa) -> None:
    settings = {**kapa.settings(), "project_id": "a/b", "base_url": "http://x.test/"}
    KapaRetriever(**settings).invoke("question")

    assert (
        str(kapa.requests[0].url) == "http://x.test/query/v1/projects/a%2Fb/retrieval/"
    )


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAPA_API_KEY", "from-env")
    monkeypatch.setenv("KAPA_PROJECT_ID", "env-project")

    retriever = KapaRetriever()

    assert retriever.api_key.get_secret_value() == "from-env"
    assert retriever.project_id == "env-project"


def test_missing_credentials_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAPA_API_KEY", raising=False)

    with pytest.raises(ValueError, match="KAPA_API_KEY"):
        KapaRetriever(project_id=PROJECT_ID)


def test_secrets_stay_out_of_representations(kapa: FakeKapa) -> None:
    retriever = KapaRetriever(**kapa.settings())
    tool = KapaGetDocumentsTool(**kapa.settings())

    for component in (retriever, tool):
        assert API_KEY not in repr(component)
        assert API_KEY not in str(component)
        assert API_KEY not in str(component.model_dump())


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (400, KapaValidationError),
        (401, KapaAuthenticationError),
        (403, KapaAuthenticationError),
        (404, KapaNotFoundError),
        (413, KapaValidationError),
        (429, KapaRateLimitError),
        (500, KapaServiceError),
        (503, KapaServiceError),
    ],
)
def test_status_errors_are_distinguished(
    kapa: FakeKapa, status: int, error: type[KapaError]
) -> None:
    kapa.override = respond(httpx.Response(status, json={"detail": "nope"}))

    with pytest.raises(error, match=f"HTTP {status}: nope") as raised:
        KapaRetriever(**kapa.settings()).invoke("question")

    assert raised.value.status_code == status  # type: ignore[attr-defined]
    assert API_KEY not in str(raised.value)
    assert len(kapa.requests) == 1


def test_validation_detail_is_kept(kapa: FakeKapa) -> None:
    detail = {"top_k": ["Ensure this value is less than or equal to 15."]}
    kapa.override = respond(httpx.Response(400, json=detail))

    with pytest.raises(KapaValidationError) as raised:
        KapaRetriever(**kapa.settings(), top_k=99).invoke("question")

    assert raised.value.detail == detail
    assert "less than or equal" in str(raised.value)


def test_rate_limit_reports_retry_after(kapa: FakeKapa) -> None:
    kapa.override = respond(
        httpx.Response(429, headers={"Retry-After": "12"}, json={"detail": "slow"})
    )

    with pytest.raises(KapaRateLimitError) as raised:
        KapaRetriever(**kapa.settings()).invoke("question")

    assert raised.value.retry_after == 12.0


def test_non_json_error_body_is_truncated(kapa: FakeKapa) -> None:
    kapa.override = respond(httpx.Response(502, text="<html>" + "x" * 2000))

    with pytest.raises(KapaServiceError) as raised:
        KapaRetriever(**kapa.settings()).invoke("question")

    assert len(str(raised.value)) < 600


def test_non_json_success_is_a_response_error(kapa: FakeKapa) -> None:
    kapa.override = respond(httpx.Response(200, text="not json"))

    with pytest.raises(KapaResponseError):
        KapaRetriever(**kapa.settings()).invoke("question")


def test_timeout_is_a_connection_error(kapa: FakeKapa) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    kapa.override = handler

    with pytest.raises(KapaConnectionError, match="timed out after 5s"):
        KapaRetriever(**kapa.settings(), timeout=5).invoke("question")

    assert len(kapa.requests) == 1


async def test_async_transport_failure_is_a_connection_error(kapa: FakeKapa) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    kapa.override = handler

    with pytest.raises(KapaConnectionError, match="ConnectError"):
        await KapaRetriever(**kapa.settings()).ainvoke("question")


def test_timeout_is_sent_with_each_request(kapa: FakeKapa) -> None:
    KapaRetriever(**kapa.settings(), timeout=7.5).invoke("question")

    assert kapa.requests[0].extensions["timeout"]["read"] == 7.5


def test_default_clients_are_created_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    kapa = FakeKapa(chunks=[{"source_url": "https://a.test", "content": "A"}])
    created: list[httpx.Client] = []
    original = httpx.Client

    def make_client(*args: object, **kwargs: object) -> httpx.Client:
        client = original(transport=httpx.MockTransport(kapa.handler))
        created.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", make_client)
    retriever = KapaRetriever(api_key=API_KEY, project_id=PROJECT_ID)

    retriever.invoke("one")
    retriever.invoke("two")

    assert len(created) == 2
    assert all(client.is_closed for client in created)
