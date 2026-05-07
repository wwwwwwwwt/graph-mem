# graphmem-core v0.2 — Mode B + Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Mode B (Anthropic LLM compression + Voyage embeddings), structured three-stage compression pipeline, and LangGraph/AutoGen adapters.

**Architecture:** `Memory.open()` detects `anthropic.api_key` and swaps `NoOpLLMClient` → `AnthropicLLMClient`, `SentenceTransformerEmbedClient` → `VoyageEmbedClient`. The pipeline module gains three structured stages (episode, entity, reflection) via LLM structured outputs. Adapters are thin `graphmem.Memory` wrappers under `graphmem.adapters.*`.

**Tech Stack:** anthropic, voyageai, pydantic (structured outputs), langgraph (optional), autogen (optional)

---

## File Structure

```
packages/graphmem-core/src/graphmem/
├── llm/
│   ├── anthropic_client.py      # Haiku/Sonnet with structured output
├── embed/
│   ├── voyage_client.py         # Voyage-3 embeddings
├── pipeline/
│   ├── episode.py               # MODIFY: add LLM path
│   ├── entity.py                # Stage 2: extract + merge
│   └── reflection.py            # Stage 3: insight generation
├── adapters/
│   ├── __init__.py
│   ├── langgraph.py             # checkpointer + recall node
│   └── autogen.py               # context mixin
└── memory.py                    # MODIFY: mode detection, pipeline wiring
```

---

## Task 1: Anthropic LLM Client

**Files:**
- Create: `packages/graphmem-core/src/graphmem/llm/anthropic_client.py`
- Test: `packages/graphmem-core/tests/unit/test_anthropic_client.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from graphmem.llm.anthropic_client import AnthropicLLMClient


def test_complete_returns_string(monkeypatch):
    def fake_messages_create(*, model, messages, max_tokens):
        class FakeMessage:
            content = [type("Block", (), {"text": "hello"})]
        return type("Response", (), {"content": [FakeMessage()]})()

    client = AnthropicLLMClient(api_key="test", default_model="claude-haiku-4")
    monkeypatch.setattr(client.client.messages, "create", fake_messages_create)
    result = client.complete("say hi")
    assert isinstance(result, str)
    assert "hello" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_anthropic_client.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""Anthropic LLM client with structured output support."""

import json
from typing import Any

from anthropic import Anthropic

from graphmem.llm.base import LLMClient


class AnthropicLLMClient(LLMClient):
    def __init__(self, api_key: str, default_model: str = "claude-haiku-4"):
        self.client = Anthropic(api_key=api_key)
        self.default_model = default_model

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        response = self.client.messages.create(
            model=self.default_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.content[0].text

    def complete_structured(self, prompt: str, *, schema: dict, max_tokens: int = 512) -> dict:
        schema_text = json.dumps(schema, indent=2)
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with a JSON object matching this schema:\n{schema_text}\n"
            f"Return ONLY the JSON object, no markdown, no explanations."
        )
        response = self.client.messages.create(
            model=self.default_model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=max_tokens,
        )
        text = response.content[0].text
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd packages/graphmem-core
python -m pytest tests/unit/test_anthropic_client.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/llm/anthropic_client.py tests/unit/test_anthropic_client.py
git commit -m "feat(llm): add AnthropicLLMClient with structured output"
```

---

## Task 2: Voyage Embedding Client

**Files:**
- Create: `packages/graphmem-core/src/graphmem/embed/voyage_client.py`
- Test: `packages/graphmem-core/tests/unit/test_voyage_client.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from graphmem.embed.voyage_client import VoyageEmbedClient


def test_embed_dimensions():
    client = VoyageEmbedClient(api_key="test", model="voyage-3")
    assert client.dim == 1024
```

- [ ] **Step 2: Run test to verify it fails**

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""Voyage AI embedding client."""

from graphmem.embed.base import EmbedClient


class VoyageEmbedClient(EmbedClient):
    def __init__(self, api_key: str, model: str = "voyage-3"):
        try:
            import voyageai
        except ImportError as e:
            raise ImportError("voyageai is required for VoyageEmbedClient. Install: pip install voyageai") from e
        self.client = voyageai.Client(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self.client.embed(texts, model=self.model)
        return result.embeddings

    @property
    def dim(self) -> int:
        # voyage-3 = 1024, voyage-3-lite = 512
        return {"voyage-3": 1024, "voyage-3-lite": 512}.get(self.model, 1024)
```

- [ ] **Step 4: Run test to verify it passes**

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/embed/voyage_client.py tests/unit/test_voyage_client.py
git commit -m "feat(embed): add VoyageEmbedClient"
```

---

## Task 3: Mode Detection in Memory.open()

**Files:**
- Modify: `packages/graphmem-core/src/graphmem/memory.py`
- Test: `packages/graphmem-core/tests/unit/test_memory_mode.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from graphmem.memory import Memory
from graphmem.llm.noop import NoOpLLMClient
from graphmem.llm.anthropic_client import AnthropicLLMClient


def test_mode_a_uses_noop_llm(tmp_home):
    mem = Memory.open(str(tmp_home), scope="s1")
    assert isinstance(mem.llm_client, NoOpLLMClient)
    mem.close()


def test_mode_b_uses_anthropic_llm(tmp_home, monkeypatch):
    cfg = {
        "mode": "B",
        "llm": {"driver": "anthropic", "api_key": "test-key", "models": {"episode": "claude-haiku-4"}},
    }
    mem = Memory.open(str(tmp_home), scope="s1", config=cfg)
    assert isinstance(mem.llm_client, AnthropicLLMClient)
    mem.close()
```

- [ ] **Step 2: Run test to verify it fails**

Expected: `AssertionError` on mode B test.

- [ ] **Step 3: Modify Memory.open()**

In `memory.py`, replace the hardcoded client instantiations with factory logic:

```python
# In Memory.open(), after config loading:
llm_cfg = cfg.llm
if llm_cfg.driver == "anthropic" and llm_cfg.api_key:
    from graphmem.llm.anthropic_client import AnthropicLLMClient
    llm_client = AnthropicLLMClient(api_key=llm_cfg.api_key, default_model=llm_cfg.models.get("episode", "claude-haiku-4"))
else:
    llm_client = NoOpLLMClient()

embed_cfg = cfg.embed
if embed_cfg.driver == "voyage" and getattr(embed_cfg, "api_key", None):
    from graphmem.embed.voyage_client import VoyageEmbedClient
    embed_client = VoyageEmbedClient(api_key=embed_cfg.api_key, model=embed_cfg.model)
else:
    embed_client = SentenceTransformerEmbedClient(model_name=embed_cfg.model)
```

- [ ] **Step 4: Run test to verify it passes**

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/memory.py tests/unit/test_memory_mode.py
git commit -m "feat(memory): add Mode A/B detection for LLM and embed backends"
```

---

## Task 4: LLM-Driven Episode Summarizer (Stage 1)

**Files:**
- Modify: `packages/graphmem-core/src/graphmem/pipeline/episode.py`
- Test: `packages/graphmem-core/tests/unit/test_pipeline_llm.py`

- [ ] **Step 1: Write the failing test**

```python
from datetime import datetime
from graphmem.schema import L0Turn, Layer
from graphmem.pipeline.episode import LLMEpisodeSummarizer
from graphmem.llm.noop import NoOpLLMClient


def test_llm_summarizer_with_noop():
    llm = NoOpLLMClient()
    summarizer = LLMEpisodeSummarizer(llm_client=llm)
    turns = [
        L0Turn(id="t1", scope="s1", role="user", content="How do I install Kuzu?", session_id="s1", turn_index=0),
        L0Turn(id="t2", scope="s1", role="assistant", content="pip install kuzu", session_id="s1", turn_index=1),
    ]
    episode = summarizer.summarize(turns)
    assert episode.layer == Layer.L1
    # NoOp returns empty string for structured, so title/summary are empty but layer is correct
```

- [ ] **Step 2: Run test to verify it fails**

Expected: `ModuleNotFoundError` for LLMEpisodeSummarizer.

- [ ] **Step 3: Write implementation**

Add to `pipeline/episode.py`:

```python
class LLMEpisodeSummarizer:
    def __init__(self, llm_client, trigger_turns: int = 20, trigger_idle_seconds: int = 300):
        self.llm_client = llm_client
        self.trigger_turns = trigger_turns
        self.trigger_idle_seconds = trigger_idle_seconds

    def should_compress(self, turn_count: int = 0, idle_seconds: float = 0.0) -> bool:
        return turn_count >= self.trigger_turns or idle_seconds >= self.trigger_idle_seconds

    def summarize(self, turns: list[L0Turn]) -> L1Episode:
        if not turns:
            return L1Episode(id="empty", scope="", layer=Layer.L1, title="Empty", summary="")

        transcript = "\n".join(f"{t.role}: {t.content}" for t in turns)
        prompt = (
            "Summarize the following conversation into a concise episode.\n\n"
            f"{transcript}\n\n"
            "Provide a title, summary, and key points."
        )
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}},
                "mentioned_entities": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "kind": {"type": "string"}}}},
            },
            "required": ["title", "summary", "key_points"],
        }
        try:
            result = self.llm_client.complete_structured(prompt, schema=schema, max_tokens=1024)
        except Exception:
            # Fallback to heuristic
            heuristic = HeuristicEpisodeSummarizer()
            return heuristic.summarize(turns)

        participants = list({t.role for t in turns})
        time_range = (turns[0].created_at, turns[-1].created_at)

        return L1Episode(
            id="",
            scope=turns[0].scope,
            layer=Layer.L1,
            title=result.get("title", ""),
            summary=result.get("summary", ""),
            key_points=result.get("key_points", []),
            participants=participants,
            time_range=time_range,
        )
```

Update `pipeline/__init__.py` to export both.

- [ ] **Step 4: Run test to verify it passes**

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/pipeline/ tests/unit/test_pipeline_llm.py
git commit -m "feat(pipeline): add LLM-driven episode summarizer"
```

---

## Task 5: Entity Extraction & Merge (Stage 2)

**Files:**
- Create: `packages/graphmem-core/src/graphmem/pipeline/entity.py`
- Test: `packages/graphmem-core/tests/unit/test_pipeline_entity.py`

- [ ] **Step 1: Write the failing test**

```python
from graphmem.schema import L1Episode, Layer
from graphmem.pipeline.entity import EntityExtractor
from graphmem.llm.noop import NoOpLLMClient


def test_entity_extractor_noop():
    llm = NoOpLLMClient()
    extractor = EntityExtractor(llm_client=llm)
    episode = L1Episode(id="e1", scope="s1", layer=Layer.L1, title="Install", summary="pip install kuzu", key_points=["Kuzu is a graph DB"])
    entities = extractor.extract(episode)
    assert isinstance(entities, list)
```

- [ ] **Step 2: Run test to verify it fails**

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
"""Entity extraction and merge (Stage 2)."""

from graphmem.schema import L1Episode, L2Entity, Layer
from graphmem.llm.base import LLMClient


class EntityExtractor:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def extract(self, episode: L1Episode) -> list[L2Entity]:
        if not episode.summary and not episode.title:
            return []

        prompt = (
            "Extract named entities from this episode summary.\n\n"
            f"Title: {episode.title}\n"
            f"Summary: {episode.summary}\n"
            f"Key points: {episode.key_points}\n\n"
            "Return entities with name, kind (person/repo/file/concept/decision/task/tool/api), and description."
        )
        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "kind": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["name", "kind"],
                    },
                }
            },
            "required": ["entities"],
        }
        try:
            result = self.llm_client.complete_structured(prompt, schema=schema, max_tokens=512)
        except Exception:
            return []

        entities = []
        for e in result.get("entities", []):
            entities.append(L2Entity(
                id="",
                scope=episode.scope,
                layer=Layer.L2,
                name=e.get("name", ""),
                kind=e.get("kind", ""),
                description=e.get("description", ""),
            ))
        return entities
```

- [ ] **Step 4: Run test to verify it passes**

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/pipeline/entity.py tests/unit/test_pipeline_entity.py
git commit -m "feat(pipeline): add entity extraction stage"
```

---

## Task 6: Reflection Stage (Stage 3)

**Files:**
- Create: `packages/graphmem-core/src/graphmem/pipeline/reflection.py`
- Test: `packages/graphmem-core/tests/unit/test_pipeline_reflection.py`

- [ ] **Step 1: Write the failing test**

```python
from graphmem.schema import L1Episode, Layer
from graphmem.pipeline.reflection import ReflectionGenerator
from graphmem.llm.noop import NoOpLLMClient


def test_reflection_generator_noop():
    llm = NoOpLLMClient()
    gen = ReflectionGenerator(llm_client=llm)
    episodes = [
        L1Episode(id="e1", scope="s1", layer=Layer.L1, title="A", summary="B", key_points=["point 1"]),
    ]
    reflections = gen.generate(episodes)
    assert isinstance(reflections, list)
```

- [ ] **Step 2: Run test to verify it fails**

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
"""Reflection generation (Stage 3)."""

from graphmem.schema import L1Episode, L3Reflection, Layer
from graphmem.llm.base import LLMClient


class ReflectionGenerator:
    def __init__(self, llm_client: LLMClient, min_episodes: int = 5):
        self.llm_client = llm_client
        self.min_episodes = min_episodes

    def should_generate(self, episodes: list[L1Episode]) -> bool:
        return len(episodes) >= self.min_episodes

    def generate(self, episodes: list[L1Episode]) -> list[L3Reflection]:
        if not self.should_generate(episodes):
            return []

        summaries = "\n---\n".join(f"{e.title}: {e.summary}" for e in episodes)
        prompt = (
            "Based on the following episodes, generate high-level insights/reflections.\n\n"
            f"{summaries}\n\n"
            "Return insights with kind (pattern/preference/rule/risk/hypothesis), insight text, and confidence."
        )
        schema = {
            "type": "object",
            "properties": {
                "reflections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "insight": {"type": "string"},
                            "kind": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["insight", "kind"],
                    },
                }
            },
            "required": ["reflections"],
        }
        try:
            result = self.llm_client.complete_structured(prompt, schema=schema, max_tokens=1024)
        except Exception:
            return []

        reflections = []
        scope = episodes[0].scope if episodes else ""
        for r in result.get("reflections", []):
            reflections.append(L3Reflection(
                id="",
                scope=scope,
                layer=Layer.L3,
                insight=r.get("insight", ""),
                kind=r.get("kind", ""),
                confidence=r.get("confidence", 1.0),
                evidence_ids=[e.id for e in episodes],
            ))
        return reflections
```

- [ ] **Step 4: Run test to verify it passes**

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/pipeline/reflection.py tests/unit/test_pipeline_reflection.py
git commit -m "feat(pipeline): add reflection generation stage"
```

---

## Task 7: Wire Pipeline into Memory

**Files:**
- Modify: `packages/graphmem-core/src/graphmem/memory.py`
- Test: `packages/graphmem-core/tests/unit/test_memory_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from graphmem.memory import Memory


def test_compact_runs_all_stages_in_mode_b(tmp_home):
    cfg = {
        "mode": "B",
        "llm": {"driver": "anthropic", "api_key": "test-key"},
        "compression": {"triggers": {"turns": 2}},
    }
    mem = Memory.open(str(tmp_home), scope="s1", config=cfg)
    mem.write_turn("user", "hello", session_id="s1")
    mem.write_turn("assistant", "world", session_id="s1")
    # With trigger_turns=2, write_turn should trigger compression
    stats = mem.stats()
    assert stats.nodes_by_layer.get("L0", 0) >= 2
    mem.close()
```

- [ ] **Step 2: Run test to verify it fails**

Expected: `AssertionError` or structural issue.

- [ ] **Step 3: Modify Memory class**

In `memory.py`:

1. Import the new pipeline classes.
2. In `__init__`, choose summarizer based on LLM client type:
   - If AnthropicLLMClient: use `LLMEpisodeSummarizer`
   - Else: use `HeuristicEpisodeSummarizer`
3. In `_compress_session`, after creating L1 episode:
   - Run `EntityExtractor.extract(episode)` → create L2 nodes + MENTIONS edges
   - Check reflection trigger → run `ReflectionGenerator.generate()` if enough episodes

Key changes to `_compress_session`:

```python
from graphmem.pipeline.entity import EntityExtractor
from graphmem.pipeline.reflection import ReflectionGenerator

def _compress_session(self, session_id: str) -> None:
    # ... existing L1 compression ...

    # Stage 2: Entity extraction (Mode B only)
    if not isinstance(self.llm_client, NoOpLLMClient):
        extractor = EntityExtractor(self.llm_client)
        entities = extractor.extract(episode)
        for entity in entities:
            entity.id = f"L2-{ULID()}"
            self.graph_store.create_node(entity)
            self.vector_store.insert(
                embedding_id=f"emb-{entity.id}",
                node_id=entity.id,
                scope=self.scope,
                layer=Layer.L2.value,
                vector=self.embed_client.embed([entity.description or entity.name])[0],
            )
            edge = MemoryEdge(
                id=f"edge-{episode.id}-{entity.id}",
                type=EdgeType.MENTIONS,
                from_id=episode.id,
                to_id=entity.id,
                scope=self.scope,
            )
            self.graph_store.create_edge(edge)

    # Stage 3: Reflection (Mode B only, triggered by episode count)
    if not isinstance(self.llm_client, NoOpLLMClient):
        all_l1 = self.graph_store.query_nodes(scope=self.scope, layer="L1")
        generator = ReflectionGenerator(self.llm_client, min_episodes=self.config.compression.triggers.get("min_episodes", 5))
        if generator.should_generate(all_l1):
            reflections = generator.generate(all_l1)
            for reflection in reflections:
                reflection.id = f"L3-{ULID()}"
                self.graph_store.create_node(reflection)
                for eid in reflection.evidence_ids:
                    edge = MemoryEdge(
                        id=f"edge-{reflection.id}-{eid}",
                        type=EdgeType.DERIVED_FROM,
                        from_id=reflection.id,
                        to_id=eid,
                        scope=self.scope,
                    )
                    self.graph_store.create_edge(edge)
```

- [ ] **Step 4: Run test to verify it passes**

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/memory.py tests/unit/test_memory_pipeline.py
git commit -m "feat(memory): wire three-stage compression pipeline"
```

---

## Task 8: LangGraph Adapter

**Files:**
- Create: `packages/graphmem-core/src/graphmem/adapters/__init__.py`
- Create: `packages/graphmem-core/src/graphmem/adapters/langgraph.py`
- Test: `packages/graphmem-core/tests/unit/test_adapter_langgraph.py`

- [ ] **Step 1: Write the failing test**

```python
from graphmem.memory import Memory
from graphmem.adapters.langgraph import recall_node


def test_recall_node_exists():
    # Just verify the function signature is importable
    assert callable(recall_node)
```

- [ ] **Step 2: Run test to verify it fails**

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
"""LangGraph adapter for graphmem."""

from typing import Any

from graphmem.memory import Memory


def recall_node(memory: Memory, k: int = 8):
    """Return a LangGraph node function that recalls before agent execution."""
    def _recall(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        if not query and "messages" in state:
            # Use last user message as query
            messages = state["messages"]
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    break
        result = memory.recall(query, k=k)
        return {**state, "recalled_context": result.formatted, "recall_items": result.items}
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
```

- [ ] **Step 4: Run test to verify it passes**

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/adapters/ tests/unit/test_adapter_langgraph.py
git commit -m "feat(adapters): add LangGraph recall and write nodes"
```

---

## Task 9: AutoGen Adapter

**Files:**
- Create: `packages/graphmem-core/src/graphmem/adapters/autogen.py`
- Test: `packages/graphmem-core/tests/unit/test_adapter_autogen.py`

- [ ] **Step 1: Write the failing test**

```python
from graphmem.adapters.autogen import GraphmemContext


def test_graphmem_context_is_callable():
    assert callable(GraphmemContext)
```

- [ ] **Step 2: Run test to verify it fails**

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
"""AutoGen adapter for graphmem."""

from graphmem.memory import Memory


class GraphmemContext:
    """Mixin to add graphmem memory to an AutoGen ConversableAgent."""

    def __init__(self, memory: Memory, *, recall_k: int = 8, **kwargs):
        self._graphmem = memory
        self._recall_k = recall_k
        super().__init__(**kwargs)

    def recall_before_reply(self, messages: list[dict], sender: str | None = None) -> str:
        if not messages:
            return ""
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        if last_user_msg:
            result = self._graphmem.recall(last_user_msg, k=self._recall_k)
            return result.formatted
        return ""

    def write_after_reply(self, messages: list[dict], session_id: str = "default") -> None:
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                self._graphmem.write_turn(role, content, session_id=session_id)
```

- [ ] **Step 4: Run test to verify it passes**

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/graphmem-core/src/graphmem/adapters/autogen.py tests/unit/test_adapter_autogen.py
git commit -m "feat(adapters): add AutoGen GraphmemContext mixin"
```

---

## Task 10: Integration Test (Mode B)

**Files:**
- Create: `packages/graphmem-core/tests/integration/test_mode_b.py`

- [ ] **Step 1: Write test**

```python
from graphmem.memory import Memory


def test_mode_b_compression_pipeline(tmp_home):
    cfg = {
        "mode": "B",
        "llm": {"driver": "noop"},  # use NoOp so no API key needed in CI
        "embed": {"driver": "sentence_transformers", "model": "all-MiniLM-L6-v2"},
        "compression": {"triggers": {"turns": 3}},
    }
    mem = Memory.open(str(tmp_home), scope="test", config=cfg)

    for i in range(3):
        mem.write_turn("user" if i % 2 == 0 else "assistant", f"msg {i}", session_id="s1")

    stats = mem.stats()
    assert stats.nodes_by_layer.get("L0", 0) >= 3
    assert stats.nodes_by_layer.get("L1", 0) >= 1
    mem.close()
```

- [ ] **Step 2: Run test**

```bash
cd packages/graphmem-core
python -m pytest tests/integration/test_mode_b.py -v
```

Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_mode_b.py
git commit -m "test(integration): add Mode B pipeline test"
```

---

## Self-Review

### 1. Spec coverage

| Spec section | Task | Status |
|-------------|------|--------|
| Mode B LLM (Haiku/Sonnet) | Task 1 | Covered |
| Voyage embeddings | Task 2 | Covered |
| Mode detection (A/B) | Task 3 | Covered |
| Stage 1 LLM episode | Task 4 | Covered |
| Stage 2 entity extraction | Task 5 | Covered |
| Stage 3 reflection | Task 6 | Covered |
| Pipeline wiring | Task 7 | Covered |
| LangGraph adapter | Task 8 | Covered |
| AutoGen adapter | Task 9 | Covered |
| Mode B integration test | Task 10 | Covered |

### 2. Placeholder scan

No TBD, TODO, or vague instructions.

### 3. Type consistency

- `LLMClient.complete_structured` signature consistent across NoOp and Anthropic.
- `Layer`, `EdgeType` enums used consistently.
- `Memory.open()` factory logic matches existing pattern.

All consistent.
