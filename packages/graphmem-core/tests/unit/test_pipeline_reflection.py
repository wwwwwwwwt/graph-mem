from graphmem.schema import L1Episode, Layer
from graphmem.pipeline.reflection import ReflectionGenerator
from graphmem.llm.noop import NoOpLLMClient


def test_reflection_generator_noop():
    llm = NoOpLLMClient()
    gen = ReflectionGenerator(llm_client=llm)
    episodes = [
        L1Episode(id="e1", scope="s1", layer=Layer.L1, title="A", summary="B", key_points=["point 1"]),
    ]
    reflections = gen.generate(episodes)
    assert isinstance(reflections, list)
    assert len(reflections) == 0  # below min_episodes default
