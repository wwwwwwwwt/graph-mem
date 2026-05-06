import pytest
from graphmem.schema import L0Turn, L1Episode, MemoryEdge, Layer, EdgeType
from graphmem.stores.kuzu_store import KuzuGraphStore


@pytest.fixture
def graph_store(tmp_home):
    store = KuzuGraphStore(str(tmp_home / "graph.db"))
    yield store
    store.close()


def test_create_and_get_node(graph_store):
    node = L0Turn(
        id="t1",
        scope="s1",
        role="user",
        content="hello",
        session_id="sess1",
        turn_index=0,
        tokens=1,
    )
    graph_store.create_node(node)
    found = graph_store.get_node("t1")
    assert found is not None
    assert found.id == "t1"
    assert found.role == "user"


def test_create_edge_and_get_neighbors(graph_store):
    n1 = L0Turn(id="t1", scope="s1", role="user", content="a", session_id="s1", turn_index=0)
    n2 = L1Episode(id="e1", scope="s1", title="ep", summary="sum")
    graph_store.create_node(n1)
    graph_store.create_node(n2)
    edge = MemoryEdge(
        id="edge1",
        type=EdgeType.DERIVED_FROM,
        from_id="e1",
        to_id="t1",
        scope="s1",
    )
    graph_store.create_edge(edge)
    neighbors = graph_store.get_neighbors("e1", direction="out")
    assert len(neighbors) == 1
    assert neighbors[0][0].type == EdgeType.DERIVED_FROM
    assert neighbors[0][1].id == "t1"


def test_query_nodes_by_scope(graph_store):
    n1 = L0Turn(id="t1", scope="s1", role="user", content="a", session_id="s1", turn_index=0)
    n2 = L0Turn(id="t2", scope="s2", role="user", content="b", session_id="s2", turn_index=0)
    graph_store.create_node(n1)
    graph_store.create_node(n2)
    results = graph_store.query_nodes(scope="s1")
    assert len(results) == 1
    assert results[0].id == "t1"


def test_count_nodes(graph_store):
    n1 = L0Turn(id="t1", scope="s1", role="user", content="a", session_id="s1", turn_index=0)
    n2 = L1Episode(id="e1", scope="s1", title="t", summary="s")
    graph_store.create_node(n1)
    graph_store.create_node(n2)
    counts = graph_store.count_nodes(scope="s1")
    assert counts["L0"] == 1
    assert counts["L1"] == 1
