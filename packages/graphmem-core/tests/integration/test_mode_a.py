from graphmem.memory import Memory


def test_mode_a_lifecycle(tmp_home):
    mem = Memory.open(str(tmp_home), scope="test")

    for i in range(5):
        mem.write_turn(
            "user" if i % 2 == 0 else "assistant",
            f"message {i}",
            session_id="sess1",
        )

    report = mem.compact()
    assert report.episodes_created >= 1

    result = mem.recall("message", k=5)
    assert len(result.items) > 0
    assert result.latency_ms < 10000

    episode_id = result.items[0].node.id
    sg = mem.graph(episode_id, depth=1)
    assert len(sg.nodes) >= 2
    assert any(e.type.value == "DERIVED_FROM" for e in sg.edges)

    stats = mem.stats()
    assert stats.nodes_by_layer.get("L0", 0) >= 5
    assert stats.nodes_by_layer.get("L1", 0) >= 1

    mem.close()
