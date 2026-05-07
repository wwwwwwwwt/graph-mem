from graphmem.schema import L1Episode, Layer
from graphmem.pipeline.entity import EntityExtractor
from graphmem.llm.noop import NoOpLLMClient


def test_entity_extractor_noop():
    llm = NoOpLLMClient()
    extractor = EntityExtractor(llm_client=llm)
    episode = L1Episode(
        id="e1", scope="s1", layer=Layer.L1,
        title="Install", summary="pip install kuzu", key_points=["Kuzu is a graph DB"]
    )
    entities = extractor.extract(episode)
    assert isinstance(entities, list)
