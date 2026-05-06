import numpy as np
import pytest
from graphmem.stores.numpy_store import NumpyVectorStore


@pytest.fixture
def vector_store(tmp_home):
    store = NumpyVectorStore(str(tmp_home / "vectors"), dim=3)
    yield store
    store.close()


def test_insert_and_search(vector_store):
    vec = [1.0, 0.0, 0.0]
    vector_store.insert("emb1", "node1", "scope1", "L1", vec)
    results = vector_store.search([1.0, 0.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0][0] == "node1"
    assert results[0][1] > 0.99


def test_search_filter_by_scope(vector_store):
    vector_store.insert("e1", "n1", "s1", "L1", [1.0, 0.0, 0.0])
    vector_store.insert("e2", "n2", "s2", "L1", [1.0, 0.0, 0.0])
    results = vector_store.search([1.0, 0.0, 0.0], k=10, scope="s1")
    assert len(results) == 1
    assert results[0][0] == "n1"


def test_search_filter_by_layer(vector_store):
    vector_store.insert("e1", "n1", "s1", "L0", [1.0, 0.0, 0.0])
    vector_store.insert("e2", "n2", "s1", "L1", [1.0, 0.0, 0.0])
    results = vector_store.search([1.0, 0.0, 0.0], k=10, layer="L1")
    assert len(results) == 1
    assert results[0][0] == "n2"


def test_persistence(tmp_home):
    path = str(tmp_home / "vectors2")
    store1 = NumpyVectorStore(path, dim=3)
    store1.insert("e1", "n1", "s1", "L1", [0.0, 1.0, 0.0])
    store1.close()

    store2 = NumpyVectorStore(path, dim=3)
    results = store2.search([0.0, 1.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0][0] == "n1"
    store2.close()
