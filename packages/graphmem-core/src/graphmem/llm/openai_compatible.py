"""Generic OpenAI-compatible LLM client (DeepSeek, OpenAI, Azure, etc.)."""

import json
import re
from typing import Any

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "openai is required for OpenAILLMClient. Install: pip install openai"
    ) from e

from graphmem.llm.base import LLMClient


def _fix_bare_unicode_escapes(value: Any) -> Any:
    """Some LLMs emit uXXXX without backslash; fix recursively."""
    if isinstance(value, str):
        def _repl(m):
            try:
                return chr(int(m.group(1), 16))
            except ValueError:
                return m.group(0)
        return re.sub(r'(?<!\\)u([0-9a-fA-F]{4})', _repl, value)
    if isinstance(value, list):
        return [_fix_bare_unicode_escapes(v) for v in value]
    if isinstance(value, dict):
        return {k: _fix_bare_unicode_escapes(v) for k, v in value.items()}
    return value


class OpenAILLMClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        default_model: str = "gpt-4o-mini",
        extra_params: dict | None = None,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.default_model = default_model
        self.extra_params = extra_params or {}

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            model=self.default_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            extra_body=self.extra_params,
        )
        return response.choices[0].message.content or ""

    def complete_structured(
        self, prompt: str, *, schema: dict, max_tokens: int = 512
    ) -> dict[str, Any]:
        schema_text = json.dumps(schema, indent=2)
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with a JSON object matching this schema:\n{schema_text}\n"
            f"Return ONLY the JSON object, no markdown, no explanations."
        )
        response = self.client.chat.completions.create(
            model=self.default_model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=max_tokens,
            extra_body=self.extra_params,
        )
        msg = response.choices[0].message
        text = (msg.content or "").strip()
        # DeepSeek reasoning models may emit reasoning_content that consumes
        # the max_tokens budget, leaving content empty.
        if not text and hasattr(msg, "reasoning_content") and msg.reasoning_content:
            raise RuntimeError(
                "LLM returned empty content (reasoning_content present). "
                "This usually means max_tokens was too small for a reasoning model. "
                "Try increasing max_tokens (e.g. 2048 or higher), or disable "
                "reasoning via extra_params (e.g. enable_thinking: false)."
            )
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        return _fix_bare_unicode_escapes(parsed)
