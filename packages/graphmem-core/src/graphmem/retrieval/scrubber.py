"""Secret scrubber for redacting sensitive data from recall results."""

import re


class SecretScrubber:
    """Detects and redacts common secret patterns from text."""

    # Patterns and their replacements
    PATTERNS = [
        # OpenAI / Anthropic / generic API keys
        (re.compile(r"\b(sk-[a-zA-Z0-9]{32,})\b"), "***API_KEY***"),
        # Bearer tokens
        (re.compile(r"\b(Bearer\s+[a-zA-Z0-9\-_]{20,})\b", re.IGNORECASE), "***BEARER_TOKEN***"),
        # Password assignments
        (re.compile(r"((?:password|passwd|pwd)\s*[:=]\s*)\S+", re.IGNORECASE), r"\1***PASSWORD***"),
        # Long hex tokens (32+ chars)
        (re.compile(r"\b([a-f0-9]{32,})\b", re.IGNORECASE), "***TOKEN***"),
        # AWS access key
        (re.compile(r"\b(AKIA[0-9A-Z]{16})\b"), "***AWS_KEY***"),
    ]

    def scrub(self, text: str) -> str:
        for pattern, replacement in self.PATTERNS:
            text = pattern.sub(replacement, text)
        return text
