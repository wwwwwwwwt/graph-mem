"""Entity extraction and merge (Stage 2)."""

from graphmem.llm.base import LLMClient
from graphmem.schema import L1Episode, L2Entity, Layer


class EntityExtractor:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def extract(self, episode: L1Episode) -> list[L2Entity]:
        if not episode.summary and not episode.title:
            return []

        prompt = (
            "Extract named entities from this episode summary.\n\n"
            f"Title: {episode.title}\n"
            f"Summary: {episode.summary}\n"
            f"Key points: {episode.key_points}\n\n"
            "Return entities with name, kind (person/repo/file/concept/decision/task/tool/api), and description."
        )
        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "kind": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["name", "kind"],
                    },
                }
            },
            "required": ["entities"],
        }
        try:
            result = self.llm_client.complete_structured(prompt, schema=schema, max_tokens=2048)
        except Exception:
            return []

        entities = []
        for e in result.get("entities", []):
            entities.append(
                L2Entity(
                    id="",
                    scope=episode.scope,
                    layer=Layer.L2,
                    name=e.get("name", ""),
                    kind=e.get("kind", ""),
                    description=e.get("description", ""),
                )
            )
        return entities
