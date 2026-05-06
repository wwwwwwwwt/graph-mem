"""NoOp LLM client for Mode A."""

from graphmem.llm.base import LLMClient


class NoOpLLMClient(LLMClient):
    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        return ""

    def complete_structured(self, prompt: str, *, schema: dict, max_tokens: int = 512) -> dict:
        return {}
