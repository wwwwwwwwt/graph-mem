"""Integration test for multi-hop graph expansion in recall."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import tempfile
from datetime import datetime, timezone

from graphmem.retrieval.search import SearchEngine
from graphmem.schema import EdgeType, L1Episode, L2Entity, Layer, MemoryEdge, MemoryNode
from graphmem.stores.base import VectorStore
from graphmem.stores.kuzu_store import KuzuGraphStore


class FakeVectorStore(VectorStore):
    """Returns a fixed seed so graph expansion is deterministic."""

    def insert(self, **kwargs):
        pass

    def search(self, query_vector, k=10, scope=None, layer=None):
        return [("L1-orm", 0.9)]

    def delete(self, embedding_id: str) -> None:
        pass

    def close(self):
        pass


def test_multi_hop_graph_recall():
    """Verify bidirectional multi-hop BFS across a chain of MENTIONS edges."""
    tmp = tempfile.mkdtemp()
    graph_store = KuzuGraphStore(str(Path(tmp) / "graph.kuzu"))

    # Build nodes
    backend_ep = L1Episode(
        id="L1-backend",
        scope="s",
        layer=Layer.L1,
        title="Backend",
        summary="Choose PostgreSQL",
    )
    migration_ep = L1Episode(
        id="L1-migration",
        scope="s",
        layer=Layer.L1,
        title="Migration",
        summary="Use Alembic",
    )
    orm_ep = L1Episode(
        id="L1-orm",
        scope="s",
        layer=Layer.L1,
        title="ORM",
        summary="SQLAlchemy + Alembic",
    )
    pg = L2Entity(
        id="L2-pg", scope="s", layer=Layer.L2, name="PostgreSQL", kind="database"
    )
    alembic = L2Entity(
        id="L2-alembic", scope="s", layer=Layer.L2, name="Alembic", kind="tool"
    )
    sqlalchemy = L2Entity(
        id="L2-sqlalchemy", scope="s", layer=Layer.L2, name="SQLAlchemy", kind="library"
    )

    for node in (backend_ep, migration_ep, orm_ep, pg, alembic, sqlalchemy):
        graph_store.create_node(node)

    # Build edges:
    # backend -> pg, migration -> pg, migration -> alembic, orm -> alembic, orm -> sqlalchemy
    edges = [
        ("L1-backend", "L2-pg"),
        ("L1-migration", "L2-pg"),
        ("L1-migration", "L2-alembic"),
        ("L1-orm", "L2-alembic"),
        ("L1-orm", "L2-sqlalchemy"),
    ]
    for from_id, to_id in edges:
        graph_store.create_edge(
            MemoryEdge(
                id=f"e-{from_id}-{to_id}",
                type=EdgeType.MENTIONS,
                from_id=from_id,
                to_id=to_id,
                scope="s",
            )
        )

    engine = SearchEngine(FakeVectorStore(), graph_store)

    # hops=0: only the seed
    result0 = engine.search(
        [0.0] * 384,
        k=10,
        max_hops=0,
        layers=(Layer.L1, Layer.L2),
        edge_types=[EdgeType.MENTIONS],
    )
    ids0 = {r.node.id for r in result0}
    assert ids0 == {"L1-orm"}, f"hops=0 should be only seed: {ids0}"

    # hops=1: seed's direct neighbors (alembic, sqlalchemy)
    result1 = engine.search(
        [0.0] * 384,
        k=10,
        max_hops=1,
        layers=(Layer.L1, Layer.L2),
        edge_types=[EdgeType.MENTIONS],
    )
    ids1 = {r.node.id for r in result1}
    assert ids1 == {"L1-orm", "L2-alembic", "L2-sqlalchemy"}, f"hops=1 wrong: {ids1}"

    # hops=2: + migration_ep via alembic (bidirectional incoming)
    result2 = engine.search(
        [0.0] * 384,
        k=10,
        max_hops=2,
        layers=(Layer.L1, Layer.L2),
        edge_types=[EdgeType.MENTIONS],
    )
    ids2 = {r.node.id for r in result2}
    assert "L1-migration" in ids2, f"migration_ep missing at hops=2: {ids2}"
    assert "L2-pg" not in ids2, f"pg should not be at hops=2: {ids2}"
    assert "L1-backend" not in ids2, f"backend_ep should not be at hops=2: {ids2}"

    # hops=3: + pg via migration_ep outgoing
    result3 = engine.search(
        [0.0] * 384,
        k=10,
        max_hops=3,
        layers=(Layer.L1, Layer.L2),
        edge_types=[EdgeType.MENTIONS],
    )
    ids3 = {r.node.id for r in result3}
    assert "L2-pg" in ids3, f"pg missing at hops=3: {ids3}"
    assert "L1-backend" not in ids3, f"backend_ep should not be at hops=3: {ids3}"

    # hops=4: + backend_ep via pg incoming
    result4 = engine.search(
        [0.0] * 384,
        k=10,
        max_hops=4,
        layers=(Layer.L1, Layer.L2),
        edge_types=[EdgeType.MENTIONS],
    )
    ids4 = {r.node.id for r in result4}
    assert "L1-backend" in ids4, f"backend_ep missing at hops=4: {ids4}"

    # hops=5: same as hops=4 (saturated)
    result5 = engine.search(
        [0.0] * 384,
        k=10,
        max_hops=5,
        layers=(Layer.L1, Layer.L2),
        edge_types=[EdgeType.MENTIONS],
    )
    ids5 = {r.node.id for r in result5}
    assert ids5 == ids4, f"hops=5 should equal hops=4 (saturated): {ids5}"

    graph_store.close()
