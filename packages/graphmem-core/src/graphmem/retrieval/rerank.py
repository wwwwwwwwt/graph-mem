"""Cross-encoder re-ranker for improving recall precision."""

from graphmem.schema import MemoryItem


class Reranker:
    """Lightweight cross-encoder re-ranker.

    Uses sentence-transformers CrossEncoder with a small model
    (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2, ~20MB).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cross-encoder model {model_name}. "
                "Install sentence-transformers to use re-ranking."
            ) from exc

    def rerank(
        self,
        query: str,
        candidates: list[MemoryItem],
        k: int = 8,
    ) -> list[MemoryItem]:
        if not candidates:
            return []

        # Build pairs: (query, candidate_text)
        pairs = []
        for item in candidates:
            text = self._item_to_text(item)
            pairs.append((query, text))

        scores = self.model.predict(pairs)

        # Combine with original scores (weighted blend)
        blended = []
        for item, rerank_score in zip(candidates, scores):
            # 70% re-ranker, 30% original (vector + graph + time_decay)
            final_score = rerank_score * 0.7 + item.score * 0.3
            blended.append(MemoryItem(node=item.node, score=final_score))

        blended.sort(key=lambda x: x.score, reverse=True)
        return blended[:k]

    @staticmethod
    def _item_to_text(item: MemoryItem) -> str:
        node = item.node
        parts = []
        if hasattr(node, "title") and node.title:
            parts.append(node.title)
        if hasattr(node, "summary") and node.summary:
            parts.append(node.summary)
        if hasattr(node, "name") and node.name:
            parts.append(node.name)
        if hasattr(node, "description") and node.description:
            parts.append(node.description)
        if node.content:
            parts.append(node.content)
        return "\n".join(parts) or ""
