"""FastAPI daemon for graphmem HTTP API."""

from fastapi import FastAPI
from pydantic import BaseModel

from graphmem.memory import Memory


def create_app(memory: Memory) -> FastAPI:
    app = FastAPI(title="graphmem-daemon")

    @app.post("/write")
    def write(payload: WritePayload):
        node_id = memory.write_turn(
            payload.role,
            payload.content,
            session_id=payload.session_id,
            tool_calls=payload.tool_calls,
            meta=payload.meta,
        )
        return {"node_id": node_id}

    @app.post("/recall")
    def recall(payload: RecallPayload):
        result = memory.recall(
            payload.query,
            k=payload.k,
            scope=payload.scope,
            token_budget=payload.token_budget,
        )
        return {
            "items": [
                {
                    "node_id": item.node.id,
                    "layer": item.node.layer.value,
                    "score": item.score,
                }
                for item in result.items
            ],
            "formatted": result.formatted,
            "tokens": result.tokens,
            "latency_ms": result.latency_ms,
        }

    @app.get("/status")
    def status():
        stats = memory.stats()
        return {
            "scope": stats.scope,
            "nodes_by_layer": stats.nodes_by_layer,
        }

    @app.post("/compact")
    def compact():
        report = memory.compact()
        return {"episodes_created": report.episodes_created}

    return app


class WritePayload(BaseModel):
    role: str
    content: str
    session_id: str
    tool_calls: list[dict] | None = None
    meta: dict | None = None


class RecallPayload(BaseModel):
    query: str
    k: int = 8
    scope: str = "current"
    token_budget: int = 4000
