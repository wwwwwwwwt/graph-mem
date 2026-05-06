import pytest
from graphmem.memory import Memory
from graphmem.schema import Layer


@pytest.fixture
def memory(tmp_home):
    mem = Memory.open(str(tmp_home), scope="test-scope")
    yield mem
    mem.close()


def test_write_turn(memory):
    nid = memory.write_turn("user", "hello", session_id="s1")
    assert nid is not None
    assert nid.startswith("L0-")


def test_recall(memory):
    memory.write_turn("user", "How do I install Kuzu?", session_id="s1")
    memory.write_turn("assistant", "pip install kuzu", session_id="s1")
    memory.compact(scope="test-scope")
    result = memory.recall("install Kuzu")
    assert len(result.items) > 0
    assert "Kuzu" in result.formatted or "install" in result.formatted


def test_graph(memory):
    memory.write_turn("user", "hello", session_id="s1")
    memory.write_turn("assistant", "hi there", session_id="s1")
    memory.compact(scope="test-scope")
    stats = memory.stats()
    assert stats.nodes_by_layer.get("L1", 0) > 0


def test_pin_and_unpin(memory):
    nid = memory.write_turn("user", "important", session_id="s1")
    memory.pin(nid)
    node = memory.graph_store.get_node(nid)
    assert node.pinned is True
    memory.unpin(nid)
    node = memory.graph_store.get_node(nid)
    assert node.pinned is False
