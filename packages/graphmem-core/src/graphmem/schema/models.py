from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, model_validator
from graphmem.schema.types import Layer, EdgeType


class MemoryNode(BaseModel):
    id: str
    layer: Layer
    scope: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = "api"
    tokens: int = 0
    pinned: bool = False
    ttl_at: datetime | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    content: str = ""


class L0Turn(MemoryNode):
    layer: Layer = Layer.L0
    role: str
    content: str
    tool_calls: list[dict] | None = None
    session_id: str
    turn_index: int

    @model_validator(mode="after")
    def compute_tokens(self):
        if self.tokens == 0 and self.content:
            self.tokens = max(1, len(self.content.split()))
        return self


class L1Episode(MemoryNode):
    layer: Layer = Layer.L1
    title: str = ""
    summary: str = ""
    time_range: tuple[datetime, datetime] | None = None
    key_points: list[str] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)


class L2Entity(MemoryNode):
    layer: Layer = Layer.L2
    name: str = ""
    kind: str = ""
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    confidence: float = 1.0


class L3Reflection(MemoryNode):
    layer: Layer = Layer.L3
    insight: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    kind: str = ""


class MemoryEdge(BaseModel):
    id: str
    type: EdgeType
    from_id: str
    to_id: str
    scope: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    weight: float = 1.0
    meta: dict[str, Any] = Field(default_factory=dict)


class MemoryItem(BaseModel):
    node: MemoryNode
    score: float = 0.0


class RecallResult(BaseModel):
    items: list[MemoryItem]
    formatted: str
    tokens: int
    latency_ms: int
    trace: dict | None = None


class Subgraph(BaseModel):
    nodes: list[MemoryNode]
    edges: list[MemoryEdge]
    root_id: str


class CompactReport(BaseModel):
    episodes_created: int = 0
    entities_created: int = 0
    reflections_created: int = 0
    tokens_processed: int = 0


class Stats(BaseModel):
    scope: str
    nodes_by_layer: dict[str, int] = Field(default_factory=dict)
    total_edges: int = 0
    total_tokens: int = 0
    oldest_node: datetime | None = None
    newest_node: datetime | None = None
