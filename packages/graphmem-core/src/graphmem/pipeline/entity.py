"""Entity extraction and merge (Stage 2)."""

import math

from graphmem.llm.base import LLMClient
from graphmem.schema import L1Episode, L2Entity, Layer


class EntityExtractor:
    def __init__(self, llm_client: LLMClient, embed_client=None, merge_threshold: float = 0.85):
        self.llm_client = llm_client
        self.embed_client = embed_client
        self.merge_threshold = merge_threshold

    def extract(self, episode: L1Episode, existing_entities: list[L2Entity] | None = None) -> list[L2Entity]:
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
            name = e.get("name", "").strip()
            kind = e.get("kind", "").strip()
            description = e.get("description", "")

            merged = self._try_merge(name, kind, description, existing_entities)
            if merged:
                entities.append(merged)
            else:
                entities.append(
                    L2Entity(
                        id="",
                        scope=episode.scope,
                        layer=Layer.L2,
                        name=name,
                        kind=kind,
                        description=description,
                    )
                )
        return entities

    def _try_merge(self, name: str, kind: str, description: str, existing_entities: list[L2Entity] | None) -> L2Entity | None:
        if not self.embed_client or not existing_entities:
            return None

        candidate_text = f"{name} {kind} {description}"
        candidate_vec = self.embed_client.embed([candidate_text])[0]

        best_match: L2Entity | None = None
        best_score = 0.0
        for existing in existing_entities:
            existing_text = f"{existing.name} {existing.kind} {existing.description or ''}"
            existing_vec = self.embed_client.embed([existing_text])[0]
            score = self._cosine_sim(candidate_vec, existing_vec)
            if score > best_score:
                best_score = score
                best_match = existing

        if best_match and best_score >= self.merge_threshold:
            # Update aliases if name is different
            if name.lower() != best_match.name.lower() and name not in (best_match.aliases or []):
                if not best_match.aliases:
                    best_match.aliases = []
                best_match.aliases.append(name)
            return best_match
        return None

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-10)
