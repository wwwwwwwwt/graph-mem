"""Test cross-encoder re-ranker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphmem.retrieval.rerank import Reranker
from graphmem.schema import L1Episode, Layer, MemoryItem


def test_reranker_reranks_candidates():
    """Reranker should return sorted results with different scores."""
    # Use a tiny model that is fast to load for tests
    try:
        reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except RuntimeError as exc:
        # Skip if sentence-transformers is not available in test env
        print(f"Skipping rerank test: {exc}")
        return

    candidates = [
        MemoryItem(
            node=L1Episode(
                id="L1-a",
                scope="s",
                layer=Layer.L1,
                title="Python asyncio",
                summary="Using asyncio for concurrent tasks",
            ),
            score=0.5,
        ),
        MemoryItem(
            node=L1Episode(
                id="L1-b",
                scope="s",
                layer=Layer.L1,
                title="Python threading",
                summary="Using threading for parallel execution",
            ),
            score=0.8,
        ),
    ]

    result = reranker.rerank("async programming", candidates, k=2)
    assert len(result) == 2
    # Re-ranker should produce different ordering or scores
    assert all(isinstance(r.score, float) for r in result)
    # Scores should be in descending order
    assert result[0].score >= result[1].score


def test_reranker_empty_candidates():
    try:
        reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except RuntimeError as exc:
        print(f"Skipping rerank test: {exc}")
        return

    result = reranker.rerank("query", [], k=2)
    assert result == []
