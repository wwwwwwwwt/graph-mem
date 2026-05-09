"""Test secret scrubber for redacting sensitive data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphmem.retrieval.scrubber import SecretScrubber


def test_scrub_openai_key():
    scrubber = SecretScrubber()
    text = "api_key = sk-abcdefghijklmnopqrstuvwxyz12345678901234567890"
    result = scrubber.scrub(text)
    assert "sk-" not in result
    assert "***API_KEY***" in result


def test_scrub_bearer_token():
    scrubber = SecretScrubber()
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    result = scrubber.scrub(text)
    assert "Bearer" not in result or "***BEARER_TOKEN***" in result


def test_scrub_password():
    scrubber = SecretScrubber()
    text = "password = mysecret123"
    result = scrubber.scrub(text)
    assert "mysecret123" not in result
    assert "***PASSWORD***" in result


def test_scrub_hex_token():
    scrubber = SecretScrubber()
    text = "token = a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
    result = scrubber.scrub(text)
    assert "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4" not in result
    assert "***TOKEN***" in result


def test_no_false_positives():
    scrubber = SecretScrubber()
    text = "The project uses Python 3.11 and FastAPI."
    result = scrubber.scrub(text)
    assert result == text


def test_scrub_multiple_secrets():
    scrubber = SecretScrubber()
    text = (
        "api_key = sk-abcdefghijklmnopqrstuvwxyz12345678901234567890\n"
        "password = secret123\n"
        "token = a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
    )
    result = scrubber.scrub(text)
    assert "***API_KEY***" in result
    assert "***PASSWORD***" in result
    assert "***TOKEN***" in result
    assert "secret123" not in result
