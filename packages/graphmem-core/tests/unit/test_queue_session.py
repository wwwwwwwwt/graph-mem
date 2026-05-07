"""Test SQLite queue session grouping."""

import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphmem.queue.sqlite_queue import SQLiteTaskQueue


def test_dequeue_by_session_groups_pending():
    db = os.path.join(tempfile.mkdtemp(), "q.db")
    q = SQLiteTaskQueue(db)
    q.enqueue("compress", {"session_id": "s1", "node_id": "L0-1"})
    q.enqueue("compress", {"session_id": "s1", "node_id": "L0-2"})
    q.enqueue("compress", {"session_id": "s2", "node_id": "L0-3"})

    groups = q.dequeue_by_session(batch_size=10)
    assert "s1" in groups
    assert "s2" in groups
    assert len(groups["s1"]) == 2
    assert len(groups["s2"]) == 1


def test_count_pending():
    db = os.path.join(tempfile.mkdtemp(), "q.db")
    q = SQLiteTaskQueue(db)
    assert q.count_pending() == 0
    q.enqueue("compress", {"session_id": "s1"})
    assert q.count_pending() == 1
    q.dequeue_by_session(batch_size=10)
    assert q.count_pending() == 0
