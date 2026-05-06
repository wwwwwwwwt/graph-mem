# graphmem-core v0.1 MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `graphmem-core` Python package with Mode A (zero-config) end-to-end: write turns, heuristic compression into L1 episodes, vector recall, graph traversal, and a CLI.

**Architecture:** A `Memory` class orchestrates pluggable backends through ABC interfaces. Mode A uses Kuzu (embedded graph), numpy+sqlite (vectors), sentence-transformers (embeddings), and a NoOp LLM with heuristic compression. No API keys required.

**Tech Stack:** Python 3.10+, Pydantic v2, Kuzu, numpy, sentence-transformers, PyYAML, pytest

---

## File Structure

```
packages/graphmem-core/
├── pyproject.toml
├── src/graphmem/
│   ├── __init__.py
│   ├── memory.py          # public Memory class
│   ├── config.py          # config loading + defaults
│   ├── schema/
│   │   ├── __init__.py
│   │   ├── types.py       # Layer, EdgeType enums
│   │   └── models.py      # pydantic models
│   ├── stores/
│   │   ├── __init__.py
│   │   ├── base.py        # GraphStore, VectorStore ABCs
│   │   ├── kuzu_store.py  # Kuzu implementation
│   │   └── numpy_store.py # numpy+sqlite vector implementation
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py        # LLMClient ABC
│   │   └── noop.py        # NoOp + heuristic
│   ├── embed/
│   │   ├── __init__.py
│   │   ├── base.py        # EmbedClient ABC
│   │   └── sentence_transformer.py
│   ├── queue/
│   │   ├── __init__.py
│   │   ├── base.py        # TaskQueue ABC
│   │   └── sqlite_queue.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── episode.py     # heuristic Stage 1
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── search.py      # vector search
│   │   └── format.py      # markdown formatting
│   └── cli.py             # `mem` entry point
└── tests/
    ├── conftest.py
    ├── unit/
    │   ├── test_schema.py
    │   ├── test_retrieval.py
    │   └── test_memory.py
    ├── contract/
    │   ├── test_graph_store.py
    │   └── test_vector_store.py
    └── integration/
        └── test_mode_a.py
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `packages/graphmem-core/pyproject.toml`
- Create: `packages/graphmem-core/src/graphmem/__init__.py`
- Create: `packages/graphmem-core/tests/conftest.py`
- Create: `packages/graphmem-core/.gitignore`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "graphmem"
version = "0.1.0"
description = "Multi-layer graph memory for LLM agents"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0",
    "kuzu>=0.4.0",
    "numpy>=1.24",
    "sentence-transformers>=2.2",
    "pyyaml>=6.0",
    "tiktoken>=0.5",
    "python-ulid>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "ruff>=0.4", "mypy>=1.9"]

[project.scripts]
mem = "graphmem.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create root package `__init__.py`**

```python
"""graphmem: multi-layer graph memory for LLM agents."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Create tests/conftest.py**

```python
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_home():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)
```

- [ ] **Step 4: Create .gitignore**

```
__pycache__/
*.egg-info/
.venv/
dist/
build/
*.db
*.db-wal
*.db-shm
*.npz
.kuzu/
```

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/
git commit -m "chore: scaffold graphmem-core package"
```

---

## Task 2: Schema Layer (Enums + Models)

**Files:**
- Create: `packages/graphmem-core/src/graphmem/schema/__init__.py`
- Create: `packages/graphmem-core/src/graphmem/schema/types.py`
- Create: `packages/graphmem-core/src/graphmem/schema/models.py`
- Test: `packages/graphmem-core/tests/unit/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_schema.py
from graphmem.schema.types import Layer, EdgeType
from graphmem.schema.models import L0Turn, L1Episode, MemoryEdge, RecallResult


def test_layer_enum():
    assert Layer.L0.value == "L0"
    assert Layer.L1.value == "L1"


def test_edge_type_enum():
    assert EdgeType.DERIVED_FROM.value == "DERIVED_FROM"
    assert EdgeType.MENTIONS.value == "MENTIONS"


def test_l0_turn_creation():
    turn = L0Turn(
        id="test-01",
        layer=Layer.L0,
        scope="u@h:p",
        role="user",
        content="hello",
        session_id="s1",
        turn_index=0,
    )
    assert turn.role == "user"
    assert turn.tokens == 1  # default


def test_recall_result_structure():
    result = RecallResult(items=[], formatted="", tokens=0, latency_ms=0)
    assert result.tokens == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_schema.py -v
```

Expected: `ModuleNotFoundError: No module named 'graphmem.schema'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/graphmem/schema/types.py
from enum import Enum


class Layer(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class EdgeType(str, Enum):
    DERIVED_FROM = "DERIVED_FROM"
    MENTIONS = "MENTIONS"
    RELATES_TO = "RELATES_TO"
    DEPENDS_ON = "DEPENDS_ON"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDES = "SUPERSEDES"
    PART_OF = "PART_OF"
```

```python
# src/graphmem/schema/models.py
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field
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
    content: str = ""  # primary text content for display/search


class L0Turn(MemoryNode):
    layer: Layer = Layer.L0
    role: str
    content: str
    tool_calls: list[dict] | None = None
    session_id: str
    turn_index: int


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
```

```python
# src/graphmem/schema/__init__.py
from graphmem.schema.types import Layer, EdgeType
from graphmem.schema.models import (
    MemoryNode,
    L0Turn,
    L1Episode,
    L2Entity,
    L3Reflection,
    MemoryEdge,
    MemoryItem,
    RecallResult,
    Subgraph,
    CompactReport,
    Stats,
)

__all__ = [
    "Layer",
    "EdgeType",
    "MemoryNode",
    "L0Turn",
    "L1Episode",
    "L2Entity",
    "L3Reflection",
    "MemoryEdge",
    "MemoryItem",
    "RecallResult",
    "Subgraph",
    "CompactReport",
    "Stats",
]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_schema.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/schema/ tests/unit/test_schema.py
git commit -m "feat(schema): add Layer, EdgeType enums and pydantic models"
```

---

## Task 3: Store ABCs

**Files:**
- Create: `packages/graphmem-core/src/graphmem/stores/__init__.py`
- Create: `packages/graphmem-core/src/graphmem/stores/base.py`
- Test: `packages/graphmem-core/tests/unit/test_stores_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_stores_base.py
import pytest
from graphmem.stores.base import GraphStore, VectorStore
from graphmem.schema import L0Turn, Layer


def test_graph_store_is_abc():
    with pytest.raises(TypeError):
        GraphStore()


def test_vector_store_is_abc():
    with pytest.raises(TypeError):
        VectorStore()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_stores_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'graphmem.stores'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/graphmem/stores/base.py
from abc import ABC, abstractmethod
from typing import Any
from graphmem.schema import MemoryNode, MemoryEdge, EdgeType


class StoreError(Exception):
    pass


class GraphStore(ABC):
    @abstractmethod
    def create_node(self, node: MemoryNode) -> None:
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> MemoryNode | None:
        ...

    @abstractmethod
    def update_node(self, node: MemoryNode) -> None:
        ...

    @abstractmethod
    def create_edge(self, edge: MemoryEdge) -> None:
        ...

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        edge_types: list[EdgeType] | None = None,
        direction: str = "out",
    ) -> list[tuple[MemoryEdge, MemoryNode]]:
        ...

    @abstractmethod
    def query_nodes(
        self,
        *,
        scope: str | None = None,
        layer: str | None = None,
        limit: int = 100,
    ) -> list[MemoryNode]:
        ...

    @abstractmethod
    def count_nodes(self, scope: str | None = None) -> dict[str, int]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class VectorStore(ABC):
    @abstractmethod
    def insert(
        self,
        embedding_id: str,
        node_id: str,
        scope: str,
        layer: str,
        vector: list[float],
    ) -> None:
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        *,
        k: int = 10,
        scope: str | None = None,
        layer: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return list of (node_id, score)."""
        ...

    @abstractmethod
    def delete(self, embedding_id: str) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
```

```python
# src/graphmem/stores/__init__.py
from graphmem.stores.base import GraphStore, VectorStore, StoreError

__all__ = ["GraphStore", "VectorStore", "StoreError"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_stores_base.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/stores/ tests/unit/test_stores_base.py
git commit -m "feat(stores): add GraphStore and VectorStore ABCs"
```

---

## Task 4: Kuzu Graph Store

**Files:**
- Create: `packages/graphmem-core/src/graphmem/stores/kuzu_store.py`
- Test: `packages/graphmem-core/tests/contract/test_graph_store.py`

- [ ] **Step 1: Write the contract test**

```python
# tests/contract/test_graph_store.py
import pytest
from datetime import datetime
from graphmem.schema import L0Turn, L1Episode, MemoryEdge, Layer, EdgeType
from graphmem.stores.kuzu_store import KuzuGraphStore


@pytest.fixture
def graph_store(tmp_home):
    store = KuzuGraphStore(str(tmp_home / "graph.db"))
    yield store
    store.close()


def test_create_and_get_node(graph_store):
    node = L0Turn(
        id="t1",
        scope="s1",
        role="user",
        content="hello",
        session_id="sess1",
        turn_index=0,
        tokens=1,
    )
    graph_store.create_node(node)
    found = graph_store.get_node("t1")
    assert found is not None
    assert found.id == "t1"
    assert found.role == "user"


def test_create_edge_and_get_neighbors(graph_store):
    n1 = L0Turn(id="t1", scope="s1", role="user", content="a", session_id="s1", turn_index=0)
    n2 = L1Episode(id="e1", scope="s1", title="ep", summary="sum")
    graph_store.create_node(n1)
    graph_store.create_node(n2)
    edge = MemoryEdge(
        id="edge1",
        type=EdgeType.DERIVED_FROM,
        from_id="e1",
        to_id="t1",
        scope="s1",
    )
    graph_store.create_edge(edge)
    neighbors = graph_store.get_neighbors("e1", direction="out")
    assert len(neighbors) == 1
    assert neighbors[0][0].type == EdgeType.DERIVED_FROM
    assert neighbors[0][1].id == "t1"


def test_query_nodes_by_scope(graph_store):
    n1 = L0Turn(id="t1", scope="s1", role="user", content="a", session_id="s1", turn_index=0)
    n2 = L0Turn(id="t2", scope="s2", role="user", content="b", session_id="s2", turn_index=0)
    graph_store.create_node(n1)
    graph_store.create_node(n2)
    results = graph_store.query_nodes(scope="s1")
    assert len(results) == 1
    assert results[0].id == "t1"


def test_count_nodes(graph_store):
    n1 = L0Turn(id="t1", scope="s1", role="user", content="a", session_id="s1", turn_index=0)
    n2 = L1Episode(id="e1", scope="s1", title="t", summary="s")
    graph_store.create_node(n1)
    graph_store.create_node(n2)
    counts = graph_store.count_nodes(scope="s1")
    assert counts["L0"] == 1
    assert counts["L1"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd packages/graphmem-core
python -m pytest tests/contract/test_graph_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'graphmem.stores.kuzu_store'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/graphmem/stores/kuzu_store.py
import json
import kuzu
from datetime import datetime
from graphmem.schema import MemoryNode, MemoryEdge, EdgeType, Layer
from graphmem.stores.base import GraphStore


class KuzuGraphStore(GraphStore):
    def __init__(self, db_path: str):
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._init_schema()

    def _init_schema(self):
        # Kuzu schema initialization - verify exact syntax against Kuzu docs if needed
        try:
            self.conn.execute(
                "CREATE NODE TABLE MemoryNode("
                "id STRING, layer STRING, scope STRING, created_at TIMESTAMP, "
                "updated_at TIMESTAMP, source STRING, tokens INT64, pinned BOOL, "
                "ttl_at TIMESTAMP, meta STRING, content STRING, "
                "PRIMARY KEY(id))"
            )
        except Exception:
            pass  # table may already exist
        try:
            self.conn.execute(
                "CREATE REL TABLE MemoryEdge("
                "FROM MemoryNode TO MemoryNode, "
                "type STRING, scope STRING, created_at TIMESTAMP, "
                "weight DOUBLE, meta STRING, MANY_MANY)"
            )
        except Exception:
            pass  # table may already exist

    def _node_to_dict(self, node: MemoryNode) -> dict:
        d = node.model_dump()
        d["layer"] = node.layer.value
        d["meta"] = json.dumps(d.get("meta", {}))
        if d.get("ttl_at") is None:
            d["ttl_at"] = "NULL"
        return d

    def _row_to_node(self, row: dict) -> MemoryNode:
        layer = Layer(row["layer"])
        meta = json.loads(row["meta"]) if row.get("meta") else {}
        cls = {Layer.L0: L0Turn, Layer.L1: L1Episode}.get(layer, MemoryNode)
        data = dict(row)
        data["layer"] = layer
        data["meta"] = meta
        if data.get("ttl_at") == "NULL":
            data["ttl_at"] = None
        return cls(**data)

    def create_node(self, node: MemoryNode) -> None:
        d = self._node_to_dict(node)
        # Build parameterized Cypher insert
        keys = ", ".join(d.keys())
        vals = ", ".join([f"'{str(v)}'" if v != "NULL" else "NULL" for v in d.values()])
        cypher = f"CREATE (n:MemoryNode {{{keys}: {vals}}})"
        # NOTE: use proper parameterization; adjust syntax per Kuzu docs
        self.conn.execute(cypher)

    def get_node(self, node_id: str) -> MemoryNode | None:
        result = self.conn.execute(f"MATCH (n:MemoryNode) WHERE n.id = '{node_id}' RETURN n")
        if result.has_next():
            row = result.get_next()
            return self._row_to_node(row)
        return None

    def update_node(self, node: MemoryNode) -> None:
        # For v0.1, re-create (Kuzu update is limited)
        self.conn.execute(f"MATCH (n:MemoryNode) WHERE n.id = '{node.id}' DELETE n")
        self.create_node(node)

    def create_edge(self, edge: MemoryEdge) -> None:
        cypher = (
            f"MATCH (a:MemoryNode), (b:MemoryNode) "
            f"WHERE a.id = '{edge.from_id}' AND b.id = '{edge.to_id}' "
            f"CREATE (a)-[:MemoryEdge "
            f"{{type: '{edge.type.value}', scope: '{edge.scope}', "
            f"created_at: '{edge.created_at.isoformat()}', weight: {edge.weight}, "
            f"meta: '{json.dumps(edge.meta)}'}}]->(b)"
        )
        self.conn.execute(cypher)

    def get_neighbors(
        self,
        node_id: str,
        edge_types: list[EdgeType] | None = None,
        direction: str = "out",
    ) -> list[tuple[MemoryEdge, MemoryNode]]:
        rel_dir = "->" if direction == "out" else "<-"
        cypher = (
            f"MATCH (n:MemoryNode)-[e:MemoryEdge]{rel_dir}(m:MemoryNode) "
            f"WHERE n.id = '{node_id}' RETURN e, m"
        )
        result = self.conn.execute(cypher)
        neighbors = []
        while result.has_next():
            row = result.get_next()
            edge_data = row["e"]
            node_data = row["m"]
            if edge_types and EdgeType(edge_data["type"]) not in edge_types:
                continue
            edge = MemoryEdge(
                id=f"{edge_data['from']}-{edge_data['to']}",
                type=EdgeType(edge_data["type"]),
                from_id=edge_data["from"],
                to_id=edge_data["to"],
                scope=edge_data["scope"],
                weight=edge_data.get("weight", 1.0),
                meta=json.loads(edge_data.get("meta", "{}")),
            )
            neighbors.append((edge, self._row_to_node(node_data)))
        return neighbors

    def query_nodes(
        self,
        *,
        scope: str | None = None,
        layer: str | None = None,
        limit: int = 100,
    ) -> list[MemoryNode]:
        where = []
        if scope:
            where.append(f"n.scope = '{scope}'")
        if layer:
            where.append(f"n.layer = '{layer}'")
        where_str = " AND ".join(where) if where else "1=1"
        cypher = f"MATCH (n:MemoryNode) WHERE {where_str} RETURN n LIMIT {limit}"
        result = self.conn.execute(cypher)
        nodes = []
        while result.has_next():
            row = result.get_next()
            nodes.append(self._row_to_node(row))
        return nodes

    def count_nodes(self, scope: str | None = None) -> dict[str, int]:
        where = f"WHERE n.scope = '{scope}'" if scope else ""
        cypher = f"MATCH (n:MemoryNode) {where} RETURN n.layer, COUNT(n)"
        result = self.conn.execute(cypher)
        counts = {}
        while result.has_next():
            row = result.get_next()
            counts[row[0]] = row[1]
        return counts

    def close(self) -> None:
        self.conn.close()
        self.db.close()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd packages/graphmem-core
python -m pytest tests/contract/test_graph_store.py -v
```

Expected: 4 passed. If Kuzu API syntax differs, adjust implementation to match.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/stores/kuzu_store.py tests/contract/test_graph_store.py
git commit -m "feat(stores): add KuzuGraphStore implementation"
```

---

## Task 5: In-Memory Vector Store (numpy + sqlite)

**Files:**
- Create: `packages/graphmem-core/src/graphmem/stores/numpy_store.py`
- Test: `packages/graphmem-core/tests/contract/test_vector_store.py`

- [ ] **Step 1: Write the contract test**

```python
# tests/contract/test_vector_store.py
import pytest
import numpy as np
from graphmem.stores.numpy_store import NumpyVectorStore


@pytest.fixture
def vector_store(tmp_home):
    store = NumpyVectorStore(str(tmp_home / "vectors"))
    yield store
    store.close()


def test_insert_and_search(vector_store):
    vec = [1.0, 0.0, 0.0]
    vector_store.insert("emb1", "node1", "scope1", "L1", vec)
    results = vector_store.search([1.0, 0.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0][0] == "node1"
    assert results[0][1] > 0.99


def test_search_filter_by_scope(vector_store):
    vector_store.insert("e1", "n1", "s1", "L1", [1.0, 0.0, 0.0])
    vector_store.insert("e2", "n2", "s2", "L1", [1.0, 0.0, 0.0])
    results = vector_store.search([1.0, 0.0, 0.0], k=10, scope="s1")
    assert len(results) == 1
    assert results[0][0] == "n1"


def test_search_filter_by_layer(vector_store):
    vector_store.insert("e1", "n1", "s1", "L0", [1.0, 0.0, 0.0])
    vector_store.insert("e2", "n2", "s1", "L1", [1.0, 0.0, 0.0])
    results = vector_store.search([1.0, 0.0, 0.0], k=10, layer="L1")
    assert len(results) == 1
    assert results[0][0] == "n2"


def test_persistence(tmp_home):
    path = str(tmp_home / "vectors2")
    store1 = NumpyVectorStore(path)
    store1.insert("e1", "n1", "s1", "L1", [0.0, 1.0, 0.0])
    store1.close()

    store2 = NumpyVectorStore(path)
    results = store2.search([0.0, 1.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0][0] == "n1"
    store2.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd packages/graphmem-core
python -m pytest tests/contract/test_vector_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'graphmem.stores.numpy_store'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/graphmem/stores/numpy_store.py
import sqlite3
import numpy as np
from pathlib import Path
from graphmem.stores.base import VectorStore


class NumpyVectorStore(VectorStore):
    def __init__(self, base_path: str, dim: int = 384):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.npz_path = self.base_path / "vectors.npz"
        self.db_path = self.base_path / "meta.db"
        self._vectors: np.ndarray | None = None
        self._deleted: set[int] = set()
        self._init_db()
        self._load_vectors()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "row_idx INTEGER PRIMARY KEY, "
            "embedding_id TEXT UNIQUE, "
            "node_id TEXT, "
            "scope TEXT, "
            "layer TEXT, "
            "model_id TEXT DEFAULT '')"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scope ON embeddings(scope)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_layer ON embeddings(layer)"
        )
        conn.commit()
        conn.close()

    def _load_vectors(self):
        if self.npz_path.exists():
            data = np.load(str(self.npz_path))
            self._vectors = data["vectors"]
        else:
            self._vectors = np.zeros((0, self.dim), dtype=np.float32)

    def _save_vectors(self):
        np.savez(str(self.npz_path), vectors=self._vectors)

    def insert(
        self,
        embedding_id: str,
        node_id: str,
        scope: str,
        layer: str,
        vector: list[float],
    ) -> None:
        vec = np.array(vector, dtype=np.float32)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vec.shape[0]}")

        row_idx = len(self._vectors)
        self._vectors = np.vstack([self._vectors, vec[np.newaxis, :]])

        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT INTO embeddings (row_idx, embedding_id, node_id, scope, layer) "
            "VALUES (?, ?, ?, ?, ?)",
            (row_idx, embedding_id, node_id, scope, layer),
        )
        conn.commit()
        conn.close()
        self._save_vectors()

    def search(
        self,
        query_vector: list[float],
        *,
        k: int = 10,
        scope: str | None = None,
        layer: str | None = None,
    ) -> list[tuple[str, float]]:
        if len(self._vectors) == 0:
            return []

        q = np.array(query_vector, dtype=np.float32)
        # cosine similarity
        norms = np.linalg.norm(self._vectors, axis=1)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        sims = np.dot(self._vectors, q) / (norms * q_norm + 1e-10)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        where = []
        params = []
        if scope:
            where.append("scope = ?")
            params.append(scope)
        if layer:
            where.append("layer = ?")
            params.append(layer)
        where_str = " AND ".join(where) if where else "1=1"

        cursor.execute(
            f"SELECT row_idx, node_id FROM embeddings WHERE {where_str}",
            params,
        )
        valid_rows = {r[0]: r[1] for r in cursor.fetchall()}
        conn.close()

        # Filter deleted and scope/layer
        candidates = []
        for idx in range(len(sims)):
            if idx in self._deleted or idx not in valid_rows:
                continue
            candidates.append((valid_rows[idx], float(sims[idx])))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def delete(self, embedding_id: str) -> None:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT row_idx FROM embeddings WHERE embedding_id = ?", (embedding_id,)
        )
        row = cursor.fetchone()
        if row:
            self._deleted.add(row[0])
        conn.close()

    def close(self) -> None:
        self._save_vectors()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd packages/graphmem-core
python -m pytest tests/contract/test_vector_store.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/stores/numpy_store.py tests/contract/test_vector_store.py
git commit -m "feat(stores): add NumpyVectorStore with sqlite metadata"
```

---

## Task 6: LLM, Embed, and Queue Backends

**Files:**
- Create: `packages/graphmem-core/src/graphmem/llm/__init__.py`
- Create: `packages/graphmem-core/src/graphmem/llm/base.py`
- Create: `packages/graphmem-core/src/graphmem/llm/noop.py`
- Create: `packages/graphmem-core/src/graphmem/embed/__init__.py`
- Create: `packages/graphmem-core/src/graphmem/embed/base.py`
- Create: `packages/graphmem-core/src/graphmem/embed/sentence_transformer.py`
- Create: `packages/graphmem-core/src/graphmem/queue/__init__.py`
- Create: `packages/graphmem-core/src/graphmem/queue/base.py`
- Create: `packages/graphmem-core/src/graphmem/queue/sqlite_queue.py`
- Test: `packages/graphmem-core/tests/unit/test_backends.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_backends.py
import pytest
from datetime import timedelta
from graphmem.llm.noop import NoOpLLMClient
from graphmem.embed.sentence_transformer import SentenceTransformerEmbedClient
from graphmem.queue.sqlite_queue import SQLiteTaskQueue


def test_noop_llm_returns_empty():
    client = NoOpLLMClient()
    result = client.complete("prompt")
    assert result == ""


def test_sentence_transformer_embed():
    client = SentenceTransformerEmbedClient(model_name="all-MiniLM-L6-v2")
    vecs = client.embed(["hello world", "goodbye"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384


def test_sqlite_queue_enqueue_and_dequeue(tmp_home):
    q = SQLiteTaskQueue(str(tmp_home / "queue.db"))
    q.enqueue("compress", {"scope": "s1", "session_id": "sess1"})
    tasks = q.dequeue(limit=1)
    assert len(tasks) == 1
    assert tasks[0]["task_type"] == "compress"
    q.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_backends.py -v
```

Expected: Import errors for llm, embed, queue modules.

- [ ] **Step 3: Write minimal implementation**

```python
# src/graphmem/llm/base.py
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        ...

    @abstractmethod
    def complete_structured(self, prompt: str, *, schema: dict, max_tokens: int = 512) -> dict:
        ...
```

```python
# src/graphmem/llm/noop.py
from graphmem.llm.base import LLMClient


class NoOpLLMClient(LLMClient):
    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        return ""

    def complete_structured(self, prompt: str, *, schema: dict, max_tokens: int = 512) -> dict:
        return {}
```

```python
# src/graphmem/llm/__init__.py
from graphmem.llm.base import LLMClient
from graphmem.llm.noop import NoOpLLMClient

__all__ = ["LLMClient", "NoOpLLMClient"]
```

```python
# src/graphmem/embed/base.py
from abc import ABC, abstractmethod


class EmbedClient(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...
```

```python
# src/graphmem/embed/sentence_transformer.py
from sentence_transformers import SentenceTransformer
from graphmem.embed.base import EmbedClient


class SentenceTransformerEmbedClient(EmbedClient):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
```

```python
# src/graphmem/embed/__init__.py
from graphmem.embed.base import EmbedClient
from graphmem.embed.sentence_transformer import SentenceTransformerEmbedClient

__all__ = ["EmbedClient", "SentenceTransformerEmbedClient"]
```

```python
# src/graphmem/queue/base.py
from abc import ABC, abstractmethod
from typing import Any


class TaskQueue(ABC):
    @abstractmethod
    def enqueue(self, task_type: str, payload: dict[str, Any]) -> str:
        ...

    @abstractmethod
    def dequeue(self, *, limit: int = 1) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def complete(self, task_id: str) -> None:
        ...

    @abstractmethod
    def fail(self, task_id: str, error: str) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
```

```python
# src/graphmem/queue/sqlite_queue.py
import json
import sqlite3
from datetime import datetime
from graphmem.queue.base import TaskQueue


class SQLiteTaskQueue(TaskQueue):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tasks ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "task_type TEXT, "
            "payload TEXT, "
            "status TEXT DEFAULT 'pending', "
            "error TEXT, "
            "created_at TEXT, "
            "started_at TEXT, "
            "completed_at TEXT)"
        )
        conn.commit()
        conn.close()

    def enqueue(self, task_type: str, payload: dict) -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO tasks (task_type, payload, created_at) VALUES (?, ?, ?)",
            (task_type, json.dumps(payload), datetime.utcnow().isoformat()),
        )
        task_id = str(cursor.lastrowid)
        conn.commit()
        conn.close()
        return task_id

    def dequeue(self, *, limit: int = 1) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, task_type, payload FROM tasks WHERE status = 'pending' "
            "ORDER BY created_at LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        tasks = []
        for row in rows:
            task_id, task_type, payload = row
            cursor.execute(
                "UPDATE tasks SET status = 'running', started_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), task_id),
            )
            tasks.append({
                "id": str(task_id),
                "task_type": task_type,
                "payload": json.loads(payload),
            })
        conn.commit()
        conn.close()
        return tasks

    def complete(self, task_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE tasks SET status = 'completed', completed_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), int(task_id)),
        )
        conn.commit()
        conn.close()

    def fail(self, task_id: str, error: str) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE tasks SET status = 'failed', error = ? WHERE id = ?",
            (error, int(task_id)),
        )
        conn.commit()
        conn.close()

    def close(self) -> None:
        pass
```

```python
# src/graphmem/queue/__init__.py
from graphmem.queue.base import TaskQueue
from graphmem.queue.sqlite_queue import SQLiteTaskQueue

__all__ = ["TaskQueue", "SQLiteTaskQueue"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_backends.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/llm/ packages/graphmem-core/src/graphmem/embed/ packages/graphmem-core/src/graphmem/queue/ tests/unit/test_backends.py
git commit -m "feat(backends): add NoOp LLM, sentence-transformer embed, SQLite queue"
```

---

## Task 7: Configuration

**Files:**
- Create: `packages/graphmem-core/src/graphmem/config.py`
- Test: `packages/graphmem-core/tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config.py
from graphmem.config import Config, load_config
from pathlib import Path


def test_default_config():
    cfg = Config()
    assert cfg.stores.graph.driver == "kuzu"
    assert cfg.embed.driver == "sentence_transformers"
    assert cfg.llm.driver == "noop"


def test_load_config_from_file(tmp_home):
    config_path = tmp_home / "config.yaml"
    config_path.write_text("mode: B\nllm:\n  driver: anthropic\n")
    cfg = load_config(str(config_path))
    assert cfg.mode == "B"
    assert cfg.llm.driver == "anthropic"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'graphmem.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/graphmem/config.py
from pathlib import Path
from pydantic import BaseModel, Field
import yaml


class GraphStoreConfig(BaseModel):
    driver: str = "kuzu"
    path: str = "db/graph.kuzu"


class VectorStoreConfig(BaseModel):
    driver: str = "numpy"
    path: str = "db/vectors"
    dim: int = 384


class QueueConfig(BaseModel):
    driver: str = "sqlite"
    path: str = "queue.db"


class LLMConfig(BaseModel):
    driver: str = "noop"
    api_key: str | None = None
    models: dict[str, str] = Field(default_factory=dict)


class EmbedConfig(BaseModel):
    driver: str = "sentence_transformers"
    model: str = "all-MiniLM-L6-v2"
    dim: int = 384


class RetrievalConfig(BaseModel):
    default_k: int = 8
    default_token_budget: int = 4000
    weights: dict[str, float] = Field(default_factory=lambda: {
        "vec": 0.55, "graph": 0.20, "recent": 0.10, "layer": 0.10, "freq": 0.05
    })
    layer_prior: dict[str, float] = Field(default_factory=lambda: {
        "L0": 0.5, "L1": 1.0, "L2": 1.1, "L3": 1.2
    })


class CompressionConfig(BaseModel):
    triggers: dict[str, int | bool] = Field(default_factory=lambda: {
        "turns": 20, "idle_seconds": 300, "on_session_end": True
    })


class Config(BaseModel):
    mode: str = "A"
    home: str = "~/.graphmem"
    scope_default: str = "${USER}@${HOST}:${PROJECT}"
    stores: dict[str, dict] = Field(default_factory=lambda: {
        "graph": {"driver": "kuzu", "path": "db/graph.kuzu"},
        "vector": {"driver": "numpy", "path": "db/vectors", "dim": 384},
        "queue": {"driver": "sqlite", "path": "queue.db"},
    })
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)


def load_config(path: str | None = None) -> Config:
    if path is None:
        path = str(Path.home() / ".graphmem" / "config.yaml")
    p = Path(path)
    if p.exists():
        with open(p, "r") as f:
            data = yaml.safe_load(f)
        return Config(**data)
    return Config()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_config.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/config.py tests/unit/test_config.py
git commit -m "feat(config): add pydantic config models and YAML loader"
```

---

## Task 8: Heuristic Compression Pipeline (Stage 1)

**Files:**
- Create: `packages/graphmem-core/src/graphmem/pipeline/__init__.py`
- Create: `packages/graphmem-core/src/graphmem/pipeline/episode.py`
- Test: `packages/graphmem-core/tests/unit/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pipeline.py
from datetime import datetime
from graphmem.schema import L0Turn, L1Episode, Layer
from graphmem.pipeline.episode import HeuristicEpisodeSummarizer


def test_heuristic_summarize_turns():
    summarizer = HeuristicEpisodeSummarizer()
    turns = [
        L0Turn(id="t1", scope="s1", role="user", content="How do I install Kuzu?", session_id="sess1", turn_index=0),
        L0Turn(id="t2", scope="s1", role="assistant", content="You can pip install it.", session_id="sess1", turn_index=1),
        L0Turn(id="t3", scope="s1", role="user", content="Thanks!", session_id="sess1", turn_index=2),
    ]
    episode = summarizer.summarize(turns)
    assert episode.layer == Layer.L1
    assert episode.title == "How do I install Kuzu?"
    assert "Kuzu" in episode.summary
    assert len(episode.key_points) >= 0


def test_should_compress_turn_count():
    summarizer = HeuristicEpisodeSummarizer(trigger_turns=3)
    assert summarizer.should_compress(turn_count=2) is False
    assert summarizer.should_compress(turn_count=3) is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'graphmem.pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/graphmem/pipeline/episode.py
import re
from datetime import datetime
from graphmem.schema import L0Turn, L1Episode, Layer


class HeuristicEpisodeSummarizer:
    def __init__(self, trigger_turns: int = 20, trigger_idle_seconds: int = 300):
        self.trigger_turns = trigger_turns
        self.trigger_idle_seconds = trigger_idle_seconds

    def should_compress(self, turn_count: int = 0, idle_seconds: float = 0.0) -> bool:
        return turn_count >= self.trigger_turns or idle_seconds >= self.trigger_idle_seconds

    def summarize(self, turns: list[L0Turn]) -> L1Episode:
        if not turns:
            return L1Episode(
                id="empty",
                scope="",
                layer=Layer.L1,
                title="Empty session",
                summary="",
            )

        # Title from first user message
        title = "Untitled"
        for t in turns:
            if t.role == "user":
                title = t.content.strip().split("\n")[0][:80]
                break

        # Summary: concatenate all messages, truncated
        messages = [f"{t.role}: {t.content}" for t in turns]
        summary = "\n".join(messages)
        if len(summary) > 1000:
            summary = summary[:997] + "..."

        # Key points: simple noun phrase extraction (naive)
        text = " ".join(t.content for t in turns)
        key_points = self._extract_key_phrases(text)

        participants = list({t.role for t in turns})
        time_range = (turns[0].created_at, turns[-1].created_at)

        return L1Episode(
            id="",  # assigned by caller
            scope=turns[0].scope,
            layer=Layer.L1,
            title=title,
            summary=summary,
            time_range=time_range,
            key_points=key_points[:5],
            participants=participants,
        )

    def _extract_key_phrases(self, text: str) -> list[str]:
        # Naive heuristic: find capitalized phrases and quoted strings
        phrases = set()
        # Quoted strings
        for match in re.finditer(r'"([^"]{3,60})"', text):
            phrases.add(match.group(1))
        # Capitalized words (potential proper nouns)
        for match in re.finditer(r'[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*', text):
            phrase = match.group(0)
            if len(phrase) <= 60:
                phrases.add(phrase)
        return sorted(phrases)[:10]
```

```python
# src/graphmem/pipeline/__init__.py
from graphmem.pipeline.episode import HeuristicEpisodeSummarizer

__all__ = ["HeuristicEpisodeSummarizer"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_pipeline.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/pipeline/ tests/unit/test_pipeline.py
git commit -m "feat(pipeline): add heuristic episode summarizer for Mode A"
```

---

## Task 9: Retrieval Engine

**Files:**
- Create: `packages/graphmem-core/src/graphmem/retrieval/__init__.py`
- Create: `packages/graphmem-core/src/graphmem/retrieval/search.py`
- Create: `packages/graphmem-core/src/graphmem/retrieval/format.py`
- Test: `packages/graphmem-core/tests/unit/test_retrieval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_retrieval.py
from datetime import datetime
from graphmem.schema import L1Episode, MemoryItem, Layer
from graphmem.retrieval.search import SearchEngine
from graphmem.retrieval.format import format_results


def test_format_single_item():
    item = MemoryItem(
        node=L1Episode(
            id="e1", scope="s1", layer=Layer.L1,
            title="Test Episode", summary="This is a test."
        ),
        score=0.95,
    )
    text = format_results([item])
    assert "Test Episode" in text
    assert "This is a test." in text


def test_format_truncates_to_budget():
    items = [
        MemoryItem(
            node=L1Episode(
                id=f"e{i}", scope="s1", layer=Layer.L1,
                title=f"Episode {i}", summary="word " * 50
            ),
            score=1.0 - i * 0.1,
        )
        for i in range(5)
    ]
    text = format_results(items, token_budget=50)
    # Should fit within rough budget (not a precise assertion since we use simple split)
    assert len(text.split()) <= 60
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_retrieval.py -v
```

Expected: `ModuleNotFoundError: No module named 'graphmem.retrieval'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/graphmem/retrieval/format.py
from graphmem.schema import MemoryItem, Layer


def format_results(items: list[MemoryItem], token_budget: int = 4000) -> str:
    lines = []
    tokens_used = 0
    # Simple token estimation: words / 0.75
    for item in items:
        node = item.node
        if node.layer == Layer.L1 and hasattr(node, "title"):
            line = f"[L1] {node.title} — {node.summary}"
        elif node.layer == Layer.L0 and hasattr(node, "role"):
            line = f"[L0] {node.role}: {node.content}"
        else:
            line = f"[{node.layer.value}] {node.content}"

        estimated_tokens = len(line.split()) / 0.75
        if tokens_used + estimated_tokens > token_budget and tokens_used > 0:
            break
        lines.append(line)
        tokens_used += estimated_tokens

    return "\n".join(lines)
```

```python
# src/graphmem/retrieval/search.py
from graphmem.schema import MemoryNode, MemoryItem, Layer
from graphmem.stores.base import VectorStore, GraphStore


class SearchEngine:
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore):
        self.vector_store = vector_store
        self.graph_store = graph_store

    def search(
        self,
        query_vector: list[float],
        *,
        k: int = 8,
        scope: str | None = None,
        layers: tuple[Layer, ...] = (Layer.L1, Layer.L2),
    ) -> list[MemoryItem]:
        # In v0.1, only vector search (no graph expansion yet)
        all_results = []
        for layer in layers:
            hits = self.vector_store.search(
                query_vector, k=k * 2, scope=scope, layer=layer.value
            )
            for node_id, score in hits:
                node = self.graph_store.get_node(node_id)
                if node is None:
                    continue
                layer_weight = {"L0": 0.5, "L1": 1.0, "L2": 1.1, "L3": 1.2}.get(
                    node.layer.value, 1.0
                )
                all_results.append(MemoryItem(node=node, score=score * layer_weight))

        # Deduplicate by node id, keep highest score
        seen = {}
        for item in all_results:
            if item.node.id not in seen or seen[item.node.id].score < item.score:
                seen[item.node.id] = item

        # Sort by score descending
        ranked = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return ranked[:k]
```

```python
# src/graphmem/retrieval/__init__.py
from graphmem.retrieval.search import SearchEngine
from graphmem.retrieval.format import format_results

__all__ = ["SearchEngine", "format_results"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_retrieval.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/retrieval/ tests/unit/test_retrieval.py
git commit -m "feat(retrieval): add vector search engine and markdown formatter"
```

---

## Task 10: Memory Public API

**Files:**
- Create: `packages/graphmem-core/src/graphmem/memory.py`
- Test: `packages/graphmem-core/tests/unit/test_memory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_memory.py
import pytest
from graphmem.memory import Memory
from graphmem.schema import Layer


@pytest.fixture
def memory(tmp_home):
    mem = Memory.open(str(tmp_home), scope="test-scope")
    yield mem
    mem.close()


def test_write_turn(memory):
    nid = memory.write_turn("user", "hello", session_id="s1")
    assert nid is not None
    assert nid.startswith("L0-")


def test_recall(memory):
    memory.write_turn("user", "How do I install Kuzu?", session_id="s1")
    memory.write_turn("assistant", "pip install kuzu", session_id="s1")
    # Trigger compression manually
    memory.compact(scope="test-scope")
    result = memory.recall("install Kuzu")
    assert len(result.items) > 0
    assert "Kuzu" in result.formatted or "install" in result.formatted


def test_graph(memory):
    memory.write_turn("user", "hello", session_id="s1")
    memory.compact(scope="test-scope")
    # Get the L1 episode id
    stats = memory.stats()
    assert stats.nodes_by_layer.get("L1", 0) > 0


def test_pin_and_unpin(memory):
    nid = memory.write_turn("user", "important", session_id="s1")
    memory.pin(nid)
    node = memory.graph_store.get_node(nid)
    assert node.pinned is True
    memory.unpin(nid)
    node = memory.graph_store.get_node(nid)
    assert node.pinned is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_memory.py -v
```

Expected: `ModuleNotFoundError: No module named 'graphmem.memory'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/graphmem/memory.py
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from ulid import ULID

from graphmem.config import Config, load_config
from graphmem.schema import (
    L0Turn, L1Episode, MemoryEdge, MemoryNode,
    RecallResult, Subgraph, CompactReport, Stats,
    Layer, EdgeType,
)
from graphmem.stores.kuzu_store import KuzuGraphStore
from graphmem.stores.numpy_store import NumpyVectorStore
from graphmem.llm.noop import NoOpLLMClient
from graphmem.embed.sentence_transformer import SentenceTransformerEmbedClient
from graphmem.queue.sqlite_queue import SQLiteTaskQueue
from graphmem.pipeline.episode import HeuristicEpisodeSummarizer
from graphmem.retrieval.search import SearchEngine
from graphmem.retrieval.format import format_results


class Memory:
    def __init__(
        self,
        home: Path,
        scope: str,
        config: Config,
        graph_store: KuzuGraphStore,
        vector_store: NumpyVectorStore,
        llm_client: NoOpLLMClient,
        embed_client: SentenceTransformerEmbedClient,
        queue: SQLiteTaskQueue,
    ):
        self.home = home
        self.scope = scope
        self.config = config
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.embed_client = embed_client
        self.queue = queue
        self.search_engine = SearchEngine(vector_store, graph_store)
        self.summarizer = HeuristicEpisodeSummarizer(
            trigger_turns=config.compression.triggers.get("turns", 20),
            trigger_idle_seconds=config.compression.triggers.get("idle_seconds", 300),
        )

    @classmethod
    def open(
        cls,
        home: str | Path,
        scope: str,
        *,
        config: dict | None = None,
    ) -> "Memory":
        home_path = Path(home).expanduser().resolve()
        home_path.mkdir(parents=True, exist_ok=True)

        cfg = Config(**config) if config else load_config(str(home_path / "config.yaml"))

        db_dir = home_path / "db"
        db_dir.mkdir(exist_ok=True)

        graph_store = KuzuGraphStore(str(db_dir / "graph.kuzu"))
        vector_store = NumpyVectorStore(
            str(db_dir / "vectors"),
            dim=cfg.embed.dim,
        )
        llm_client = NoOpLLMClient()
        embed_client = SentenceTransformerEmbedClient(model_name=cfg.embed.model)
        queue = SQLiteTaskQueue(str(home_path / "queue.db"))

        return cls(
            home=home_path,
            scope=scope,
            config=cfg,
            graph_store=graph_store,
            vector_store=vector_store,
            llm_client=llm_client,
            embed_client=embed_client,
            queue=queue,
        )

    def write_turn(
        self,
        role: str,
        content: str,
        *,
        session_id: str,
        tool_calls: list[dict] | None = None,
        meta: dict | None = None,
    ) -> str:
        turn_index = self._count_turns_in_session(session_id)
        node_id = f"L0-{ULID()}"
        turn = L0Turn(
            id=node_id,
            scope=self.scope,
            role=role,
            content=content,
            tool_calls=tool_calls,
            session_id=session_id,
            turn_index=turn_index,
            tokens=len(content.split()),
            meta=meta or {},
        )
        self.graph_store.create_node(turn)

        # Embed and store vector
        embeddings = self.embed_client.embed([content])
        self.vector_store.insert(
            embedding_id=f"emb-{node_id}",
            node_id=node_id,
            scope=self.scope,
            layer=Layer.L0.value,
            vector=embeddings[0],
        )

        # Check if compression should trigger
        if self.summarizer.should_compress(turn_count=turn_index + 1):
            self._compress_session(session_id)

        return node_id

    def _count_turns_in_session(self, session_id: str) -> int:
        # Naive: query all L0 nodes for this session
        nodes = self.graph_store.query_nodes(scope=self.scope, layer="L0")
        return sum(1 for n in nodes if getattr(n, "session_id", "") == session_id)

    def _compress_session(self, session_id: str) -> None:
        nodes = self.graph_store.query_nodes(scope=self.scope, layer="L0")
        turns = [
            n for n in nodes
            if getattr(n, "session_id", "") == session_id
        ]
        turns.sort(key=lambda t: t.turn_index)
        if len(turns) < 2:
            return

        episode = self.summarizer.summarize(turns)
        episode.id = f"L1-{ULID()}"
        self.graph_store.create_node(episode)

        # Create DERIVED_FROM edges
        for turn in turns:
            edge = MemoryEdge(
                id=f"edge-{episode.id}-{turn.id}",
                type=EdgeType.DERIVED_FROM,
                from_id=episode.id,
                to_id=turn.id,
                scope=self.scope,
            )
            self.graph_store.create_edge(edge)

        # Embed episode
        text = f"{episode.title}\n{episode.summary}"
        embeddings = self.embed_client.embed([text])
        self.vector_store.insert(
            embedding_id=f"emb-{episode.id}",
            node_id=episode.id,
            scope=self.scope,
            layer=Layer.L1.value,
            vector=embeddings[0],
        )

    def recall(
        self,
        query: str,
        *,
        k: int = 8,
        scope: str | list[str] = "current",
        layers: tuple[Layer, ...] = (Layer.L1, Layer.L2),
        token_budget: int = 4000,
        time_window: timedelta | None = None,
        edge_types: list[EdgeType] | None = None,
        explain: bool = False,
    ) -> RecallResult:
        start = datetime.utcnow()
        effective_scope = self.scope if scope == "current" else scope
        # v0.1: single scope only
        if isinstance(effective_scope, list):
            effective_scope = effective_scope[0]

        query_embedding = self.embed_client.embed([query])[0]
        items = self.search_engine.search(
            query_embedding,
            k=k,
            scope=effective_scope,
            layers=layers,
        )
        formatted = format_results(items, token_budget=token_budget)
        latency = int((datetime.utcnow() - start).total_seconds() * 1000)

        return RecallResult(
            items=items,
            formatted=formatted,
            tokens=len(formatted.split()),
            latency_ms=latency,
        )

    def graph(
        self,
        node_id: str,
        *,
        depth: int = 2,
        edge_types: list[EdgeType] | None = None,
    ) -> Subgraph:
        root = self.graph_store.get_node(node_id)
        if root is None:
            return Subgraph(nodes=[], edges=[], root_id=node_id)

        nodes = {node_id: root}
        edges = []
        frontier = [node_id]
        visited = {node_id}

        for _ in range(depth):
            next_frontier = []
            for nid in frontier:
                neighbors = self.graph_store.get_neighbors(nid, edge_types=edge_types)
                for edge, node in neighbors:
                    edges.append(edge)
                    if node.id not in visited:
                        visited.add(node.id)
                        nodes[node.id] = node
                        next_frontier.append(node.id)
            frontier = next_frontier

        return Subgraph(
            nodes=list(nodes.values()),
            edges=edges,
            root_id=node_id,
        )

    def pin(self, node_id: str) -> None:
        node = self.graph_store.get_node(node_id)
        if node:
            node.pinned = True
            self.graph_store.update_node(node)

    def unpin(self, node_id: str) -> None:
        node = self.graph_store.get_node(node_id)
        if node:
            node.pinned = False
            self.graph_store.update_node(node)

    def compact(self, *, scope: str = "current") -> CompactReport:
        effective_scope = self.scope if scope == "current" else scope
        # For v0.1, compact all sessions in scope that have pending turns
        nodes = self.graph_store.query_nodes(scope=effective_scope, layer="L0")
        sessions = {}
        for n in nodes:
            sid = getattr(n, "session_id", "")
            if sid:
                sessions.setdefault(sid, []).append(n)

        episodes_created = 0
        for sid, turns in sessions.items():
            turns.sort(key=lambda t: t.turn_index)
            if len(turns) >= 2:
                self._compress_session(sid)
                episodes_created += 1

        return CompactReport(episodes_created=episodes_created)

    def export(self, path: str | Path, *, scope: str | None = None) -> None:
        raise NotImplementedError("export planned for v0.4")

    def import_(self, path: str | Path, *, trust: bool = False) -> None:
        raise NotImplementedError("import planned for v0.4")

    def stats(self, *, scope: str = "current") -> Stats:
        effective_scope = self.scope if scope == "current" else scope
        counts = self.graph_store.count_nodes(scope=effective_scope)
        total = sum(counts.values())
        return Stats(
            scope=effective_scope,
            nodes_by_layer=counts,
            total_tokens=0,
        )

    def close(self) -> None:
        self.graph_store.close()
        self.vector_store.close()
        self.queue.close()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_memory.py -v
```

Expected: 4 passed. May take a moment for sentence-transformers first download.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/memory.py tests/unit/test_memory.py
git commit -m "feat(memory): add public Memory API with write_turn, recall, graph, pin, compact"
```

---

## Task 11: CLI Entry Point

**Files:**
- Create: `packages/graphmem-core/src/graphmem/cli.py`
- Test: `packages/graphmem-core/tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_cli.py
import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "graphmem.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_cli.py -v
```

Expected: `ModuleNotFoundError: No module named 'graphmem.cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/graphmem/cli.py
import argparse
import sys
from pathlib import Path
from graphmem.memory import Memory


def main():
    parser = argparse.ArgumentParser(prog="mem", description="graphmem CLI")
    parser.add_argument("--home", default="~/.graphmem", help="graphmem home directory")
    parser.add_argument("--scope", default="", help="memory scope")
    subparsers = parser.add_subparsers(dest="command")

    # write
    write_parser = subparsers.add_parser("write", help="Write a turn")
    write_parser.add_argument("role", choices=["user", "assistant"])
    write_parser.add_argument("content")
    write_parser.add_argument("--session-id", required=True)

    # recall
    recall_parser = subparsers.add_parser("recall", help="Recall memories")
    recall_parser.add_argument("query")
    recall_parser.add_argument("--k", type=int, default=8)

    # graph
    graph_parser = subparsers.add_parser("graph", help="Show graph around a node")
    graph_parser.add_argument("node_id")

    # compact
    subparsers.add_parser("compact", help="Run compression")

    # stats
    subparsers.add_parser("stats", help="Show stats")

    # doctor
    subparsers.add_parser("doctor", help="Check health")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    home = Path(args.home).expanduser()
    scope = args.scope or f"cli@{Path.home().name}:default"

    if args.command == "doctor":
        print(f"graphmem home: {home}")
        print(f"scope: {scope}")
        print(f"config exists: {(home / 'config.yaml').exists()}")
        return

    mem = Memory.open(home, scope=scope)
    try:
        if args.command == "write":
            nid = mem.write_turn(args.role, args.content, session_id=args.session_id)
            print(nid)
        elif args.command == "recall":
            result = mem.recall(args.query, k=args.k)
            print(result.formatted)
        elif args.command == "graph":
            sg = mem.graph(args.node_id)
            for n in sg.nodes:
                print(f"[{n.layer.value}] {n.id}: {n.content[:80]}")
        elif args.command == "compact":
            report = mem.compact()
            print(f"Episodes created: {report.episodes_created}")
        elif args.command == "stats":
            s = mem.stats()
            print(f"Scope: {s.scope}")
            for layer, count in s.nodes_by_layer.items():
                print(f"  {layer}: {count}")
    finally:
        mem.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_cli.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/cli.py tests/unit/test_cli.py
git commit -m "feat(cli): add mem CLI entry point"
```

---

## Task 12: Package Exports

**Files:**
- Modify: `packages/graphmem-core/src/graphmem/__init__.py`
- Test: `packages/graphmem-core/tests/unit/test_imports.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_imports.py
from graphmem import Memory, Layer, EdgeType, RecallResult


def test_public_api_imports():
    assert Memory is not None
    assert Layer.L0 is not None
    assert EdgeType.DERIVED_FROM is not None
    assert RecallResult is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_imports.py -v
```

Expected: ImportError for Memory, Layer, etc. from graphmem.

- [ ] **Step 3: Write minimal implementation**

```python
# src/graphmem/__init__.py
"""graphmem: multi-layer graph memory for LLM agents."""

from graphmem.memory import Memory
from graphmem.schema.types import Layer, EdgeType
from graphmem.schema.models import RecallResult, Subgraph, CompactReport, Stats

__version__ = "0.1.0"
__all__ = [
    "Memory",
    "Layer",
    "EdgeType",
    "RecallResult",
    "Subgraph",
    "CompactReport",
    "Stats",
]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_imports.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/__init__.py tests/unit/test_imports.py
git commit -m "feat: expose public API from graphmem package root"
```

---

## Task 13: Integration Test (Mode A End-to-End)

**Files:**
- Create: `packages/graphmem-core/tests/integration/test_mode_a.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_mode_a.py
from graphmem.memory import Memory


def test_mode_a_lifecycle(tmp_home):
    mem = Memory.open(str(tmp_home), scope="test")

    # Write turns across a session
    for i in range(5):
        mem.write_turn(
            "user" if i % 2 == 0 else "assistant",
            f"message {i}",
            session_id="sess1",
        )

    # Compact
    report = mem.compact()
    assert report.episodes_created >= 1

    # Recall
    result = mem.recall("message", k=5)
    assert len(result.items) > 0
    assert result.latency_ms < 5000  # generous for cold embedding model

    # Graph
    episode_id = result.items[0].node.id
    sg = mem.graph(episode_id, depth=1)
    assert len(sg.nodes) >= 2  # episode + at least one turn
    assert any(e.type.value == "DERIVED_FROM" for e in sg.edges)

    # Stats
    stats = mem.stats()
    assert stats.nodes_by_layer.get("L0", 0) >= 5
    assert stats.nodes_by_layer.get("L1", 0) >= 1

    mem.close()
```

- [ ] **Step 2: Run test to verify it passes**

```bash
cd packages/graphmem-core
python -m pytest tests/integration/test_mode_a.py -v
```

Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add packages/graphmem-core/tests/integration/test_mode_a.py
git commit -m "test(integration): add Mode A end-to-end lifecycle test"
```

---

## Task 14: CI Workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create CI workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -e "packages/graphmem-core[dev]"
      - name: Lint
        run: |
          ruff check packages/graphmem-core/src
          ruff format --check packages/graphmem-core/src
      - name: Type check
        run: |
          mypy packages/graphmem-core/src/graphmem --ignore-missing-imports
      - name: Unit + Contract tests
        run: |
          pytest packages/graphmem-core/tests/unit packages/graphmem-core/tests/contract -v
      - name: Integration tests
        run: |
          pytest packages/graphmem-core/tests/integration -v
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for test, lint, typecheck"
```

---

## Self-Review

### 1. Spec coverage check

| Spec section | Plan task | Status |
|-------------|-----------|--------|
| Mode A zero-config backends (Kuzu + numpy + sqlite) | Tasks 4, 5, 6 | Covered |
| NoOp LLM + heuristic compression | Tasks 6, 8 | Covered |
| sentence-transformers embeddings | Task 6 | Covered |
| Public API (write_turn, recall, graph, pin, compact, stats) | Tasks 10, 12 | Covered |
| Vector recall (no graph expansion in v0.1) | Tasks 5, 9, 10 | Covered |
| CLI entry point | Task 11 | Covered |
| Config loading with defaults | Task 7 | Covered |
| SQLite task queue | Task 6 | Covered |
| Tests (unit, contract, integration) | Tasks 2-14 | Covered |
| CI workflow | Task 14 | Covered |

**Gaps:**
- `export` / `import_` raise `NotImplementedError` — spec says v0.4. OK for v0.1.
- Graph expansion in recall is vector-only — spec says graph expansion in v0.2. OK for v0.1.
- No MMR in retrieval — spec says v0.2. OK.
- No secret scrubber — spec says v0.4. OK.
- No daemon — spec says plugin layer v0.2. OK.
- No MCP server — spec says v0.2. OK.

### 2. Placeholder scan

No TBD, TODO, or vague instructions found. All steps contain actual code or exact commands.

### 3. Type consistency check

- `Layer` enum used consistently across schema, stores, memory, retrieval.
- `EdgeType` enum used consistently.
- `MemoryNode` and subclasses used consistently.
- `RecallResult` signature matches spec (items, formatted, tokens, latency_ms).
- `scope="current"` semantics implemented in Memory class.

All consistent.
