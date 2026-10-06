"""Fictional Python client for the Fernwick Relay API, used as evaluation material."""

import hashlib
import hmac
import random

DEFAULT_TIMEOUT_SECONDS = 10
MAX_PAYLOAD_BYTES = 262_144
SIGNATURE_HEADER = "X-Fernwick-Signature"


def backoff_delay(attempt: int) -> float:
    """Return the delay in seconds before retry number `attempt`, starting at 1."""
    base = 2.0 * (2 ** (attempt - 1))
    capped = min(base, 300.0)
    return capped * random.uniform(0.8, 1.0)


def verify_signature(secret: bytes, body: bytes, signature: str) -> bool:
    """Check an X-Fernwick-Signature header against the raw request body."""
    expected = hmac.new(secret, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


class FernwickClient:
    """Minimal client for creating relays and replaying dead letters."""

    def __init__(self, token: str, region: str = "eu-north") -> None:
        self.token = token
        self.base_url = f"https://{region}.api.fernwick.example"

    def replay_dead_letter(self, delivery_id: str) -> str:
        """Return the endpoint that replays one dead-lettered delivery."""
        return f"{self.base_url}/v2/deliveries/{delivery_id}/replay"
