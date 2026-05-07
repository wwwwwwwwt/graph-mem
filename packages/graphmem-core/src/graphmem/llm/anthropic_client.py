"""Anthropic LLM client with structured output support."""

import json
from typing import Any

try:
    from anthropic import Anthropic
except ImportError as e:
    raise ImportError(
        "anthropic is required for AnthropicLLMClient. Install: pip install anthropic"
    ) from e

from graphmem.llm.base import LLMClient


class AnthropicLLMClient(LLMClient):
    def __init__(self, api_key: str, default_model: str = "claude-haiku-4"):
        self.client = Anthropic(api_key=api_key)
        self.default_model = default_model

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        response = self.client.messages.create(
            model=self.default_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.content[0].text

    def complete_structured(
        self, prompt: str, *, schema: dict, max_tokens: int = 512
    ) -> dict[str, Any]:
        schema_text = json.dumps(schema, indent=2)
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with a JSON object matching this schema:\n{schema_text}\n"
            f"Return ONLY the JSON object, no markdown, no explanations."
        )
        response = self.client.messages.create(
            model=self.default_model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=max_tokens,
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
