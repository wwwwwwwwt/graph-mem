"""BGE Chinese embedding client."""

from graphmem.embed.base import EmbedClient


class BGEEmbedClient(EmbedClient):
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", dim: int = 512):
        self.model_name = model_name
        self._dim = dim
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. Install: pip install sentence-transformers"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    @property
    def dim(self) -> int:
        return self._dim
