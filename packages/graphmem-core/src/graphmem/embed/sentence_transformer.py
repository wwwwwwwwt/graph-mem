"""Sentence-transformers embedding client."""

from sentence_transformers import SentenceTransformer

from graphmem.embed.base import EmbedClient


class SentenceTransformerEmbedClient(EmbedClient):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
