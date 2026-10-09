import langchain_kapa_ai

EXPECTED_ALL = {
    "KapaAPIError",
    "KapaAuthenticationError",
    "KapaConnectionError",
    "KapaDocument",
    "KapaDocumentsPage",
    "KapaEndUser",
    "KapaError",
    "KapaGetDocumentsInput",
    "KapaGetDocumentsTool",
    "KapaNotFoundError",
    "KapaRateLimitError",
    "KapaResponseError",
    "KapaRetriever",
    "KapaServiceError",
    "KapaToolkit",
    "KapaValidationError",
    "__version__",
}


def test_public_api() -> None:
    assert set(langchain_kapa_ai.__all__) == EXPECTED_ALL
    for name in langchain_kapa_ai.__all__:
        assert hasattr(langchain_kapa_ai, name)
