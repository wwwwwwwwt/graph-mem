"""Vector search engine."""

from graphmem.schema import Layer, MemoryItem
from graphmem.stores.base import GraphStore, VectorStore


class SearchEngine:
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore):
        self.vector_store = vector_store
        self.graph_store = graph_store

    def search(
        self,
        query_vector: list[float],
        *,
        k: int = 8,
        scope: str | None = None,
        layers: tuple[Layer, ...] = (Layer.L1, Layer.L2),
    ) -> list[MemoryItem]:
        all_results = []
        for layer in layers:
            hits = self.vector_store.search(
                query_vector, k=k * 2, scope=scope, layer=layer.value
            )
            for node_id, score in hits:
                node = self.graph_store.get_node(node_id)
                if node is None:
                    continue
                layer_weight = {"L0": 0.5, "L1": 1.0, "L2": 1.1, "L3": 1.2}.get(
                    node.layer.value, 1.0
                )
                all_results.append(MemoryItem(node=node, score=score * layer_weight))

        seen = {}
        for item in all_results:
            if item.node.id not in seen or seen[item.node.id].score < item.score:
                seen[item.node.id] = item

        ranked = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return ranked[:k]
