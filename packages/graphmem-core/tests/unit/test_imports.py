from graphmem import Memory, Layer, EdgeType, RecallResult


def test_public_api_imports():
    assert Memory is not None
    assert Layer.L0 is not None
    assert EdgeType.DERIVED_FROM is not None
    assert RecallResult is not None
