"""LLM client abstract base class."""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        ...

    @abstractmethod
    def complete_structured(self, prompt: str, *, schema: dict, max_tokens: int = 512) -> dict:
        ...
