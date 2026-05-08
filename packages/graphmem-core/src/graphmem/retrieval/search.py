"""Vector search engine."""

import math
from datetime import datetime, timedelta, timezone

from graphmem.schema import EdgeType, Layer, MemoryItem
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
        expand_graph: bool = True,
        max_expansion_nodes: int = 20,
        max_hops: int = 2,
        edge_types: list[EdgeType] | None = None,
        time_window: timedelta | None = None,
        halflife_days: float = 30.0,
    ) -> list[MemoryItem]:
        now = datetime.now(timezone.utc)
        cutoff = now - time_window if time_window else None

        all_results = []
        for layer in layers:
            hits = self.vector_store.search(
                query_vector, k=k * 2, scope=scope, layer=layer.value
            )
            for node_id, score in hits:
                node = self.graph_store.get_node(node_id)
                if node is None:
                    continue
                if cutoff and node.created_at and node.created_at < cutoff:
                    continue
                layer_weight = {"L0": 0.5, "L1": 1.0, "L2": 1.1, "L3": 1.2}.get(
                    node.layer.value, 1.0
                )
                node_created = node.created_at
                if node_created and node_created.tzinfo is None:
                    node_created = node_created.replace(tzinfo=timezone.utc)
                age_seconds = max(0, (now - node_created).total_seconds()) if node_created else 0
                age_days = age_seconds / 86400.0
                time_decay = math.exp(-age_days / halflife_days)
                all_results.append(
                    MemoryItem(node=node, score=score * layer_weight * time_decay)
                )

        # Graph expansion: bidirectional BFS up to max_hops
        if expand_graph:
            expanded_ids = {item.node.id for item in all_results}
            hop_decay = {1: 0.7, 2: 0.5, 3: 0.35, 4: 0.25, 5: 0.18}

            # frontier: list of (node_id, base_score_from_seed)
            frontier = [(item.node.id, item.score) for item in all_results]
            hop = 0
            while frontier and hop < max_hops:
                hop += 1
                next_frontier = []
                for node_id, base_score in frontier:
                    # Bidirectional: both incoming and outgoing edges
                    for direction in ("out", "in"):
                        for edge, neighbor in self.graph_store.get_neighbors(
                            node_id, direction=direction, edge_types=edge_types
                        ):
                            if neighbor.id in expanded_ids:
                                continue
                            if len(expanded_ids) >= max_expansion_nodes + len(all_results):
                                break
                            expanded_ids.add(neighbor.id)
                            decay = hop_decay.get(hop, 0.18)
                            all_results.append(
                                MemoryItem(node=neighbor, score=base_score * decay)
                            )
                            next_frontier.append((neighbor.id, base_score * decay))
                frontier = next_frontier

        seen = {}
        for item in all_results:
            if item.node.id not in seen or seen[item.node.id].score < item.score:
                seen[item.node.id] = item

        ranked = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return ranked[:k]
