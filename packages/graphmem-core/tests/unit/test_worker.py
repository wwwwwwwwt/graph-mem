"""Test CompressionWorker."""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphmem.memory import Memory
from graphmem.worker import CompressionWorker


def test_worker_compresses_pending_turns():
    tmp = tempfile.mkdtemp()
    mem = Memory.open(tmp, scope="sw", config={
        "mode": "A",
        "compression": {"triggers": {"turns": 999, "min_episodes": 99}},
    })
    nid1 = mem.write_turn("user", "hello", session_id="sid1")
    nid2 = mem.write_turn("assistant", "hi", session_id="sid1")

    # Manually enqueue compression tasks (Task 4 will make write_turn do this)
    mem.queue.enqueue("compress", {"session_id": "sid1", "node_id": nid1})
    mem.queue.enqueue("compress", {"session_id": "sid1", "node_id": nid2})

    worker = CompressionWorker(mem)
    report = worker.run_once()
    assert report.episodes_created >= 1

    stats = mem.stats()
    assert stats.nodes_by_layer.get("L1", 0) >= 1
    mem.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_worker_skips_single_turn_session():
    tmp = tempfile.mkdtemp()
    mem = Memory.open(tmp, scope="sw2", config={
        "mode": "A",
        "compression": {"triggers": {"turns": 999, "min_episodes": 99}},
    })
    mem.write_turn("user", "only one", session_id="sid1")

    worker = CompressionWorker(mem)
    report = worker.run_once()
    assert report.episodes_created == 0
    mem.close()
    shutil.rmtree(tmp, ignore_errors=True)
