from graphmem.memory import Memory


def test_mode_b_compression_pipeline(tmp_home):
    cfg = {
        "mode": "B",
        "llm": {"driver": "noop"},
        "embed": {"driver": "sentence_transformers", "model": "all-MiniLM-L6-v2"},
        "compression": {"triggers": {"turns": 3}},
    }
    mem = Memory.open(str(tmp_home), scope="test", config=cfg)

    for i in range(3):
        mem.write_turn("user" if i % 2 == 0 else "assistant", f"msg {i}", session_id="s1")

    mem.compact(scope="test")
    stats = mem.stats()
    assert stats.nodes_by_layer.get("L0", 0) >= 3
    assert stats.nodes_by_layer.get("L1", 0) >= 1
    mem.close()
