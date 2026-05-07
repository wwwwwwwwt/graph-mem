from graphmem.schema import L0Turn, Layer
from graphmem.pipeline.episode import LLMEpisodeSummarizer
from graphmem.llm.noop import NoOpLLMClient


def test_llm_summarizer_with_noop():
    llm = NoOpLLMClient()
    summarizer = LLMEpisodeSummarizer(llm_client=llm)
    turns = [
        L0Turn(id="t1", scope="s1", role="user", content="How do I install Kuzu?", session_id="s1", turn_index=0),
        L0Turn(id="t2", scope="s1", role="assistant", content="pip install kuzu", session_id="s1", turn_index=1),
    ]
    episode = summarizer.summarize(turns)
    assert episode.layer == Layer.L1
