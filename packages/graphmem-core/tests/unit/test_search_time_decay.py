"""Test time decay and time_window filtering in SearchEngine."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import math

from graphmem.retrieval.search import SearchEngine
from graphmem.schema import Layer, MemoryItem, MemoryNode
from graphmem.stores.base import GraphStore, VectorStore


class FakeVectorStore(VectorStore):
    def insert(self, **kwargs):
        pass

    def search(self, query_vector, k=10, scope=None, layer=None):
        return [("old", 0.9), ("new", 0.9)]

    def delete(self, embedding_id: str) -> None:
        pass

    def close(self):
        pass


class FakeGraphStore(GraphStore):
    def create_node(self, node):
        pass

    def get_node(self, node_id):
        now = datetime.now(timezone.utc)
        if node_id == "old":
            return MemoryNode(
                id="old",
                scope="s",
                layer=Layer.L1,
                content="old",
                created_at=now - timedelta(days=60),
            )
        if node_id == "new":
            return MemoryNode(
                id="new",
                scope="s",
                layer=Layer.L1,
                content="new",
                created_at=now - timedelta(days=1),
            )
        return None

    def get_neighbors(self, node_id, edge_types=None, direction="out"):
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


def test_time_decay_lowers_old_node_score():
    engine = SearchEngine(FakeVectorStore(), FakeGraphStore())
    results = engine.search([0.1] * 384, k=2, scope="s", layers=(Layer.L1,))
    assert len(results) == 2
    old_score = next(r.score for r in results if r.node.id == "old")
    new_score = next(r.score for r in results if r.node.id == "new")
    # 60 days old should decay to exp(-60/30) = ~0.135
    # 1 day old should decay to exp(-1/30) = ~0.967
    assert new_score > old_score
    assert old_score < 0.2
    assert new_score > 0.8


def test_time_window_filters_old_nodes():
    engine = SearchEngine(FakeVectorStore(), FakeGraphStore())
    results = engine.search(
        [0.1] * 384,
        k=2,
        scope="s",
        layers=(Layer.L1,),
        time_window=timedelta(days=7),
    )
    ids = {r.node.id for r in results}
    assert "new" in ids
    assert "old" not in ids


def test_time_decay_formula():
    """Verify the exact decay formula."""
    engine = SearchEngine(FakeVectorStore(), FakeGraphStore())
    results = engine.search([0.1] * 384, k=2, scope="s", layers=(Layer.L1,))
    old_item = next(r for r in results if r.node.id == "old")
    new_item = next(r for r in results if r.node.id == "new")

    expected_old = 0.9 * math.exp(-60 / 30.0)
    expected_new = 0.9 * math.exp(-1 / 30.0)

    assert abs(old_item.score - expected_old) < 0.01
    assert abs(new_item.score - expected_new) < 0.01
