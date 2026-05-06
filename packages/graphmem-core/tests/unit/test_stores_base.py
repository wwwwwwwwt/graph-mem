import pytest
from graphmem.stores.base import GraphStore, VectorStore


def test_graph_store_is_abc():
    with pytest.raises(TypeError):
        GraphStore()


def test_vector_store_is_abc():
    with pytest.raises(TypeError):
        VectorStore()
