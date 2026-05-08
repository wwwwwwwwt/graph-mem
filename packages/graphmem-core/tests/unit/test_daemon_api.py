"""Test FastAPI daemon."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fastapi.testclient import TestClient
from graphmem.memory import Memory
from graphmem.daemon.main import create_app


def test_daemon_recall_endpoint():
    tmp = tempfile.mkdtemp()
    mem = Memory.open(tmp, scope="sd", config={"mode": "A"})
    mem.write_turn("user", "discussing PostgreSQL for backend", session_id="sid")
    mem.compact(scope="sd")

    app = create_app(mem)
    client = TestClient(app)
    resp = client.post("/recall", json={"query": "database", "k": 4})
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    mem.close()


def test_daemon_write_endpoint():
    tmp = tempfile.mkdtemp()
    mem = Memory.open(tmp, scope="sd", config={"mode": "A"})

    app = create_app(mem)
    client = TestClient(app)
    resp = client.post(
        "/write",
        json={"role": "user", "content": "hello", "session_id": "s1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"].startswith("L0-")
    mem.close()


def test_daemon_status_endpoint():
    tmp = tempfile.mkdtemp()
    mem = Memory.open(tmp, scope="sd", config={"mode": "A"})

    app = create_app(mem)
    client = TestClient(app)
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes_by_layer" in data
    mem.close()
