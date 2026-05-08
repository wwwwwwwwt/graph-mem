"""Test graph expansion in recall."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphmem.retrieval.search import SearchEngine
from graphmem.schema import MemoryEdge, EdgeType, Layer, MemoryItem, MemoryNode
from graphmem.stores.base import GraphStore, VectorStore


class FakeVectorStore(VectorStore):
    def insert(self, **kwargs):
        pass

    def search(self, query_vector, k=8, scope=None, layer=None):
        return [("L1-1", 0.9)]

    def delete(self, embedding_id: str) -> None:
        pass

    def close(self):
        pass


class FakeGraphStore(GraphStore):
    def create_node(self, node):
        pass

    def get_node(self, node_id):
        if node_id == "L1-1":
            return MemoryNode(id="L1-1", scope="s", layer=Layer.L1, content="episode")
        if node_id == "L2-1":
            return MemoryNode(id="L2-1", scope="s", layer=Layer.L2, content="entity")
        return None

    def get_neighbors(self, node_id, edge_types=None, direction="out"):
        if node_id == "L1-1":
            return [(
                MemoryEdge(
                    id="e1", type=EdgeType.MENTIONS,
                    from_id="L1-1", to_id="L2-1", scope="s",
                ),
                self.get_node("L2-1"),
            )]
        return []

    def query_nodes(self, **kwargs):
        return []

    def create_edge(self, edge):
        pass

    def update_node(self, node):
        pass

    def count_nodes(self, scope=None):
        return {}

    def close(self):
        pass


def test_search_expands_along_graph_edges():
    engine = SearchEngine(FakeVectorStore(), FakeGraphStore())
    results = engine.search([0.1] * 384, k=2, scope="s", layers=(Layer.L1,), expand_graph=True)
    ids = [r.node.id for r in results]
    assert "L2-1" in ids, f"Graph expansion should bring in L2 entity, got {ids}"


def test_search_no_expansion_when_disabled():
    engine = SearchEngine(FakeVectorStore(), FakeGraphStore())
    results = engine.search([0.1] * 384, k=2, scope="s", layers=(Layer.L1,), expand_graph=False)
    ids = [r.node.id for r in results]
    assert "L2-1" not in ids, f"Expansion disabled, got {ids}"
