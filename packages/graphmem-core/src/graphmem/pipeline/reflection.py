"""Reflection generation (Stage 3)."""

from graphmem.llm.base import LLMClient
from graphmem.schema import L1Episode, L3Reflection, Layer


class ReflectionGenerator:
    def __init__(self, llm_client: LLMClient, min_episodes: int = 5):
        self.llm_client = llm_client
        self.min_episodes = min_episodes

    def should_generate(self, episodes: list[L1Episode]) -> bool:
        return len(episodes) >= self.min_episodes

    def generate(self, episodes: list[L1Episode]) -> list[L3Reflection]:
        if not self.should_generate(episodes):
            return []

        summaries = "\n---\n".join(f"{e.title}: {e.summary}" for e in episodes)
        prompt = (
            "Based on the following episodes, generate high-level insights/reflections.\n\n"
            f"{summaries}\n\n"
            "Return insights with kind (pattern/preference/rule/risk/hypothesis), insight text, and confidence."
        )
        schema = {
            "type": "object",
            "properties": {
                "reflections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "insight": {"type": "string"},
                            "kind": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["insight", "kind"],
                    },
                }
            },
            "required": ["reflections"],
        }
        try:
            result = self.llm_client.complete_structured(prompt, schema=schema, max_tokens=2048)
        except Exception:
            return []

        reflections = []
        scope = episodes[0].scope if episodes else ""
        for r in result.get("reflections", []):
            reflections.append(
                L3Reflection(
                    id="",
                    scope=scope,
                    layer=Layer.L3,
                    insight=r.get("insight", ""),
                    kind=r.get("kind", ""),
                    confidence=r.get("confidence", 1.0),
                    evidence_ids=[e.id for e in episodes],
                )
            )
        return reflections
