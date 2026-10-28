import hashlib
import secrets

KEY_PREFIX = "lrag_"


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(24)


def hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
