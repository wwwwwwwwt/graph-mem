"""Test entity merge/deduplication."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphmem.pipeline.entity import EntityExtractor
from graphmem.llm.noop import NoOpLLMClient
from graphmem.schema import L1Episode, Layer
from graphmem.embed.sentence_transformer import SentenceTransformerEmbedClient


class FakeLLM(NoOpLLMClient):
    def complete_structured(self, prompt, *, schema, max_tokens=512):
        return {
            "entities": [
                {"name": "PostgreSQL", "kind": "tool", "description": "Database"},
            ]
        }


def test_entity_merge_prevents_duplicates():
    embed = SentenceTransformerEmbedClient(model_name="all-MiniLM-L6-v2")
    extractor = EntityExtractor(FakeLLM(), embed_client=embed, merge_threshold=0.85)

    ep1 = L1Episode(
        id="L1-1", scope="s", layer=Layer.L1,
        title="Backend", summary="Use PostgreSQL", key_points=[],
    )
    ep2 = L1Episode(
        id="L1-2", scope="s", layer=Layer.L1,
        title="Migration", summary="Migrate to PostgreSQL", key_points=[],
    )

    ents1 = extractor.extract(ep1)
    assert len(ents1) == 1
    assert ents1[0].name == "PostgreSQL"

    # Second extraction with existing entities should merge
    ents2 = extractor.extract(ep2, existing_entities=ents1)
    assert len(ents2) == 1
    assert ents2[0].name == "PostgreSQL"
    # Should be the same object (merged)
    assert ents2[0] is ents1[0]
