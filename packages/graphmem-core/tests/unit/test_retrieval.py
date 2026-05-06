from graphmem.schema import L1Episode, MemoryItem, Layer
from graphmem.retrieval.format import format_results


def test_format_single_item():
    item = MemoryItem(
        node=L1Episode(
            id="e1", scope="s1", layer=Layer.L1,
            title="Test Episode", summary="This is a test."
        ),
        score=0.95,
    )
    text = format_results([item])
    assert "Test Episode" in text
    assert "This is a test." in text


def test_format_truncates_to_budget():
    items = [
        MemoryItem(
            node=L1Episode(
                id=f"e{i}", scope="s1", layer=Layer.L1,
                title=f"Episode {i}", summary="word " * 50
            ),
            score=1.0 - i * 0.1,
        )
        for i in range(5)
    ]
    text = format_results(items, token_budget=50)
    assert len(text.split()) <= 60
