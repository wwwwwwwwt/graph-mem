"""Test MCP server tools."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphmem.memory import Memory
from graphmem.mcp.server import build_tools


def test_mcp_recall_tool():
    tmp = tempfile.mkdtemp()
    mem = Memory.open(tmp, scope="sm", config={"mode": "A"})
    mem.write_turn("user", "use PostgreSQL", session_id="sid")
    mem.compact(scope="sm")

    tools = build_tools(mem)
    recall_tool = next(t for t in tools if t["name"] == "recall_memory")
    result = recall_tool["handler"]({"query": "database", "k": 4})
    assert "formatted" in result
    mem.close()


def test_mcp_stats_tool():
    tmp = tempfile.mkdtemp()
    mem = Memory.open(tmp, scope="sm", config={"mode": "A"})

    tools = build_tools(mem)
    stats_tool = next(t for t in tools if t["name"] == "get_stats")
    result = stats_tool["handler"]({})
    assert "nodes_by_layer" in result
    mem.close()
