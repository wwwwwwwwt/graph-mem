"""MCP server tools for graphmem."""

from graphmem.memory import Memory


def build_tools(memory: Memory) -> list[dict]:
    return [
        {
            "name": "recall_memory",
            "description": "Retrieve relevant historical memory episodes and entities for the current context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "k": {"type": "integer", "default": 8},
                },
                "required": ["query"],
            },
            "handler": lambda params: _recall(memory, params),
        },
        {
            "name": "list_sessions",
            "description": "List recent project scopes/sessions in memory.",
            "parameters": {"type": "object", "properties": {}},
            "handler": lambda params: _list_sessions(memory),
        },
        {
            "name": "get_stats",
            "description": "Get memory system statistics.",
            "parameters": {"type": "object", "properties": {}},
            "handler": lambda params: _get_stats(memory),
        },
    ]


def _recall(memory: Memory, params: dict) -> dict:
    result = memory.recall(params["query"], k=params.get("k", 8))
    return {
        "formatted": result.formatted,
        "item_count": len(result.items),
        "latency_ms": result.latency_ms,
    }


def _list_sessions(memory: Memory) -> dict:
    stats = memory.stats()
    return {"scope": stats.scope, "nodes": stats.nodes_by_layer}


def _get_stats(memory: Memory) -> dict:
    stats = memory.stats()
    return {"scope": stats.scope, "nodes_by_layer": stats.nodes_by_layer}
