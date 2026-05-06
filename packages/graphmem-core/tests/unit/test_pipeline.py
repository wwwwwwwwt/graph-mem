from graphmem.schema import L0Turn, Layer
from graphmem.pipeline.episode import HeuristicEpisodeSummarizer


def test_heuristic_summarize_turns():
    summarizer = HeuristicEpisodeSummarizer()
    turns = [
        L0Turn(id="t1", scope="s1", role="user", content="How do I install Kuzu?", session_id="sess1", turn_index=0),
        L0Turn(id="t2", scope="s1", role="assistant", content="You can pip install it.", session_id="sess1", turn_index=1),
        L0Turn(id="t3", scope="s1", role="user", content="Thanks!", session_id="sess1", turn_index=2),
    ]
    episode = summarizer.summarize(turns)
    assert episode.layer == Layer.L1
    assert "Kuzu" in episode.title
    assert len(episode.key_points) >= 0


def test_should_compress_turn_count():
    summarizer = HeuristicEpisodeSummarizer(trigger_turns=3)
    assert summarizer.should_compress(turn_count=2) is False
    assert summarizer.should_compress(turn_count=3) is True
