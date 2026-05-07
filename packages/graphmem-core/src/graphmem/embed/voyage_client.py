"""Voyage AI embedding client."""

from graphmem.embed.base import EmbedClient


class VoyageEmbedClient(EmbedClient):
    def __init__(self, api_key: str, model: str = "voyage-3"):
        try:
            import voyageai
        except ImportError as e:
            raise ImportError(
                "voyageai is required for VoyageEmbedClient. Install: pip install voyageai"
            ) from e
        self.client = voyageai.Client(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self.client.embed(texts, model=self.model)
        return result.embeddings

    @property
    def dim(self) -> int:
        return {"voyage-3": 1024, "voyage-3-lite": 512}.get(self.model, 1024)
