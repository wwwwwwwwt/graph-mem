from graphmem.memory import Memory
from graphmem.adapters.langgraph import recall_node, write_node


def test_recall_node_callable(tmp_home):
    mem = Memory.open(str(tmp_home), scope="s1")
    node = recall_node(mem, k=4)
    assert callable(node)
    mem.close()


def test_write_node_callable(tmp_home):
    mem = Memory.open(str(tmp_home), scope="s1")
    node = write_node(mem)
    assert callable(node)
    mem.close()
