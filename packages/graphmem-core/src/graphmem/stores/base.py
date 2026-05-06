from abc import ABC, abstractmethod
from graphmem.schema import MemoryNode, MemoryEdge, EdgeType


class StoreError(Exception):
    pass


class GraphStore(ABC):
    @abstractmethod
    def create_node(self, node: MemoryNode) -> None:
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> MemoryNode | None:
        ...

    @abstractmethod
    def update_node(self, node: MemoryNode) -> None:
        ...

    @abstractmethod
    def create_edge(self, edge: MemoryEdge) -> None:
        ...

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        edge_types: list[EdgeType] | None = None,
        direction: str = "out",
    ) -> list[tuple[MemoryEdge, MemoryNode]]:
        ...

    @abstractmethod
    def query_nodes(
        self,
        *,
        scope: str | None = None,
        layer: str | None = None,
        limit: int = 100,
    ) -> list[MemoryNode]:
        ...

    @abstractmethod
    def count_nodes(self, scope: str | None = None) -> dict[str, int]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class VectorStore(ABC):
    @abstractmethod
    def insert(
        self,
        embedding_id: str,
        node_id: str,
        scope: str,
        layer: str,
        vector: list[float],
    ) -> None:
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        *,
        k: int = 10,
        scope: str | None = None,
        layer: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return list of (node_id, score)."""
        ...

    @abstractmethod
    def delete(self, embedding_id: str) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
