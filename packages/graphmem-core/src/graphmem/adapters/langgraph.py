"""LangGraph adapter for graphmem."""

from typing import Any

from graphmem.memory import Memory


def recall_node(memory: Memory, k: int = 8):
    """Return a LangGraph node function that recalls before agent execution."""

    def _recall(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        if not query and "messages" in state:
            messages = state["messages"]
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    break
        result = memory.recall(query, k=k)
        return {
            **state,
            "recalled_context": result.formatted,
            "recall_items": result.items,
        }

    return _recall


def write_node(memory: Memory):
    """Return a LangGraph node function that writes turns after agent execution."""

    def _write(state: dict[str, Any]) -> dict[str, Any]:
        session_id = state.get("session_id", "default")
        messages = state.get("messages", [])
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                memory.write_turn(role, content, session_id=session_id)
        return state

    return _write
