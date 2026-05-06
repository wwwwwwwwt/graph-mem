from graphmem.schema.types import Layer, EdgeType
from graphmem.schema.models import L0Turn, L1Episode, MemoryEdge, RecallResult


def test_layer_enum():
    assert Layer.L0.value == "L0"
    assert Layer.L1.value == "L1"


def test_edge_type_enum():
    assert EdgeType.DERIVED_FROM.value == "DERIVED_FROM"
    assert EdgeType.MENTIONS.value == "MENTIONS"


def test_l0_turn_creation():
    turn = L0Turn(
        id="test-01",
        layer=Layer.L0,
        scope="u@h:p",
        role="user",
        content="hello",
        session_id="s1",
        turn_index=0,
    )
    assert turn.role == "user"
    assert turn.tokens == 1


def test_recall_result_structure():
    result = RecallResult(items=[], formatted="", tokens=0, latency_ms=0)
    assert result.tokens == 0
