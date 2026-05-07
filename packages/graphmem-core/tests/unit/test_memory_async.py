"""Test async write_turn behavior."""

import time
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphmem.memory import Memory


def test_write_turn_does_not_block_on_compression(monkeypatch):
    tmp = tempfile.mkdtemp()
    cfg = {
        "mode": "A",
        "compression": {"triggers": {"turns": 3, "idle_seconds": 300}},
    }
    mem = Memory.open(tmp, scope="s", config=cfg)

    call_count = [0]
    original_compress = mem._compress_session
    def slow_compress(sid):
        call_count[0] += 1
        return original_compress(sid)
    monkeypatch.setattr(mem, "_compress_session", slow_compress)

    t0 = time.time()
    mem.write_turn("user", "a", session_id="sid")
    mem.write_turn("assistant", "b", session_id="sid")
    mem.write_turn("user", "c", session_id="sid")
    elapsed = time.time() - t0

    assert elapsed < 1.0, "write_turn should be fast (<1s)"
    assert call_count[0] == 0, "compression should not happen synchronously"
    mem.close()
