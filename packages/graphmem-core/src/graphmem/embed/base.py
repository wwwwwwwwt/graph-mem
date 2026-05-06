"""Embedding client abstract base class."""

from abc import ABC, abstractmethod


class EmbedClient(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...
