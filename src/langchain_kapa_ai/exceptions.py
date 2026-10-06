from __future__ import annotations

from typing import Any


class KapaError(Exception):
    """Base class for every error raised by langchain-kapa-ai."""


class KapaConnectionError(KapaError):
    """The request did not reach Kapa or timed out."""


class KapaResponseError(KapaError):
    """Kapa returned a response that does not match the expected format."""


class KapaAPIError(KapaError):
    """Kapa answered with an error status."""

    def __init__(self, message: str, *, status_code: int, detail: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class KapaAuthenticationError(KapaAPIError):
    """The API key is missing, invalid, or not valid for the project."""


class KapaNotFoundError(KapaAPIError):
    """The project does not exist for this API key."""


class KapaValidationError(KapaAPIError):
    """Kapa rejected the request parameters."""


class KapaRateLimitError(KapaAPIError):
    """A rate limit or quota was exceeded."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        detail: Any = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, detail=detail)
        self.retry_after = retry_after


class KapaServiceError(KapaAPIError):
    """Kapa failed to process the request."""
