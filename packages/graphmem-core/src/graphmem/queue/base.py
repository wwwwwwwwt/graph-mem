"""Task queue abstract base class."""

from abc import ABC, abstractmethod
from typing import Any


class TaskQueue(ABC):
    @abstractmethod
    def enqueue(self, task_type: str, payload: dict[str, Any]) -> str:
        ...

    @abstractmethod
    def dequeue(self, *, limit: int = 1) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def complete(self, task_id: str) -> None:
        ...

    @abstractmethod
    def fail(self, task_id: str, error: str) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
