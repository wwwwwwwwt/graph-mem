from graphmem.adapters.autogen import GraphmemContext


def test_graphmem_context_callable():
    assert callable(GraphmemContext)
