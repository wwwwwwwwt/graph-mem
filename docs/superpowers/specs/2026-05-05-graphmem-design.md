# graphmem Design Specification

- **Status**: Draft, awaiting user review
- **Date**: 2026-05-05
- **Author**: Brainstorming session (Claude Code + user)
- **Scope**: Full design for a graph-based, multi-layer memory system delivered as both a Claude Code plugin and a portable Python library/MCP server usable by any agent framework.

---

## 0. Goals & Non-Goals

### Goals
1. **Persistent multi-layer memory** for LLM agents: raw turns, episode summaries, semantic entities, and reflections, with a typed graph linking them.
2. **Two consumption surfaces with one core**: a Claude Code plugin (`/plugin install`) and a portable Python package usable from LangGraph, AutoGen, CrewAI, DSPy, OpenAI Agents, raw Anthropic Messages, or any process that speaks MCP / HTTP.
3. **Hybrid retrieval**: vector top-K seeds + typed graph expansion + fusion scoring, with a token budget contract.
4. **Zero-config baseline (Mode A)** that matches claude-mem's "drop-in summarization" experience without requiring any external service.
5. **Full-graph mode (Mode B)** that adds typed entities/edges, reflections, and graph-aware recall once an Anthropic API key is present.
6. **Team/production mode (Mode C)** with shared remote backends (Neo4j + pgvector / Qdrant), authn/authz, and HTTP API.

### Non-Goals (v1)
- No bespoke embedding training; we consume off-the-shelf embeddings.
- No fine-tuned compression model; we use prompted Anthropic models with structured outputs.
- No GUI; observability is logs + JSON exports + an optional CLI `mem doctor`.
- No multi-tenant SaaS in v1 (Mode C ships as self-hosted Docker compose).
- No automatic conflict-resolution beyond surfacing CONTRADICTS edges.

### Success criteria
- p95 hook overhead in Claude Code < 50 ms (daemon warm path); cold-start recall is skipped, not blocked.
- p95 `recall()` end-to-end < 400 ms with embedded backend, < 250 ms warm cache.
- Mode B reaches **parity** with claude-mem on a fixed recall benchmark; Mode A is "functional baseline" — it stores and retrieves but does not claim parity, since claude-mem uses an LLM on every tool call while Mode A is heuristic-only.
- Mode B answers "did we already solve this?" with at least one correct cross-session hit on a 30-task benchmark.
- A LangGraph quickstart works in ≤ 20 lines, end to end, against the same store as the Claude Code plugin.
- Mode B respects a configurable daily token budget; over-budget runs degrade gracefully (skip Stage 3, downsample Stage 1 inputs).

---

## 1. Architecture Overview

### 1.1 Layered model

```
┌──────────────────────────────────────────────────────────────────┐
│  Frontends (thin adapters)                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ Claude Code │  │ LangGraph / │  │ MCP server  │  │ HTTP    │ │
│  │   plugin    │  │ AutoGen /…  │  │ (stdio/sse) │  │ (v2)    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────┬────┘ │
│         │                │                │              │       │
│         ▼                ▼                ▼              ▼       │
│              ┌───────────────────────────────────┐               │
│              │         graphmem.Memory           │  ← Public API │
│              │ write / recall / graph / pin /…   │               │
│              └─────────────────┬─────────────────┘               │
└────────────────────────────────┼─────────────────────────────────┘
                                 │ depends on ABCs only
        ┌────────────┬───────────┼────────────┬────────────┐
        ▼            ▼           ▼            ▼            ▼
   GraphStore   VectorStore   LLMClient   EmbedClient  TaskQueue
   (Kuzu /     (sqlite-vss /  (Anthropic /(Voyage /     (SQLite /
    Neo4j)      Qdrant /       NoOp)      sentence-     Redis)
                pgvector)                  transformers)
```

The core library never imports a concrete backend; everything goes through abstract base classes (ABCs) under `graphmem.stores`, `graphmem.llm`, `graphmem.embed`, `graphmem.queue`. Frontends never reach below `graphmem.Memory`.

### 1.2 Operating modes

| Mode | Trigger | Backends | LLM compression | Recall quality | Install cost |
|------|---------|----------|-----------------|----------------|--------------|
| **A — Zero-config** | No API key | Kuzu + sqlite-vss + sentence-transformers (MiniLM) | NoOp + heuristic chunker | Vector + graph (sparser, heuristic-built) | `pip install graphmem` only |
| **B — Full graph** | Anthropic key present | Same as A | Haiku (L1) + Sonnet (L2/L3) | Hybrid vector+graph w/ entities & reflections | + API key |
| **C — Team/prod** | `GRAPHMEM_BACKEND=remote` | Neo4j + Qdrant/pgvector + Redis queue | Same as B | Same as B + shared scope | Docker compose / k8s |

Mode A is the **honest offline baseline**: persistence + vector recall + structural graph work, but compression quality is heuristic and known to be below LLM-driven systems (including claude-mem, which calls an LLM on every tool use). Mode B is the **parity tier** vs claude-mem and the primary experience. Mode C is opt-in for teams.

### 1.3 Process topology (Claude Code plugin path)

```
hook event (bash wrapper, <5 ms cold)
    │
    ▼
Unix socket (~/.graphmem/daemon.sock)
    │
    ▼
graphmem-daemon (Python, long-lived)
    ├─ Memory.write_turn(...)        ← synchronous, append-only
    ├─ enqueue compression task      ← non-blocking
    └─ on UserPromptSubmit:
         Memory.recall(...) → format → return text injected into prompt

graphmem-worker (background process, same venv)
    └─ loops queue.db, runs compression pipeline (Section 3)
```

### 1.4 Key invariants

- **L0 is append-only.** Compression never edits or deletes raw turns; it produces new layered nodes linked via `DERIVED_FROM`.
- **Every node carries `scope`** (flat string `user@host:project`, see Section 2.1) so a single store can host many isolated worlds.
- **Hooks must never block the user.** Recall has a hard token-and-time budget; writes go straight to L0 and an enqueue.
- **The daemon is optional.** Every entry point (CLI, MCP, SDK) can run standalone without a daemon for portability; the daemon exists only to amortize Python startup cost for shell hooks.

---

## 2. Data Model

### 2.1 Scope

`scope` is a single flat string that identifies the memory world.

- Format: `<user>@<host>:<project_id>` for personal; `team:<team_id>:<project_id>` for shared.
- Examples: `tw@mac:graphmem`, `team:acme:billing-svc`.
- Stored verbatim on every node and edge; queries always carry a scope filter.
- "Promoting" a memory to a wider scope copies the node with new scope and a `SUPERSEDES` or `PART_OF` edge to the original.

### 2.2 Node types

All nodes share these fields:

```python
{
  "id": str,              # ULID
  "layer": "L0|L1|L2|L3",
  "scope": str,
  "created_at": datetime,
  "updated_at": datetime,
  "source": "hook|cli|mcp|api|adapter:<name>",
  "embedding_id": str | None,    # foreign key to vector store
  "tokens": int,                 # measured token count
  "pinned": bool,
  "ttl_at": datetime | None,     # GC hint
  "meta": dict                   # frontend-specific payload
}
```

Layer-specific fields:

- **L0 Turn** — `role`, `content`, `tool_calls`, `session_id`, `turn_index`. Default TTL 7 days.
- **L1 Episode** — `title`, `summary`, `time_range`, `key_points: list[str]`, `participants: list[str]`. Default TTL 90 days.
- **L2 Entity** — `name`, `kind` (`person|repo|file|concept|decision|task|tool|api`), `aliases: list[str]`, `description`, `confidence: float`. Permanent.
- **L3 Reflection** — `insight`, `evidence_ids: list[str]`, `confidence: float`, `kind` (`pattern|preference|rule|risk|hypothesis`). Permanent.

### 2.3 Edge types

| Type | From → To | Semantics | Default weight |
|------|-----------|-----------|----------------|
| `DERIVED_FROM` | L1/L2/L3 → L0/L1 | Compression provenance | 1.0 |
| `MENTIONS` | L0/L1 → L2 | Turn/episode references entity | 0.7 |
| `RELATES_TO` | L2 ↔ L2 | Symmetric semantic relation | 1.0 |
| `DEPENDS_ON` | L2 → L2 | Directed dependency | 1.2 |
| `CONTRADICTS` | any ↔ any | Conflict marker | 1.3 |
| `SUPERSEDES` | new → old | Replacement (newer is canonical) | 1.5 |
| `PART_OF` | child → parent | Hierarchy / scope promotion | 1.0 |

Edge fields: `id`, `type`, `from_id`, `to_id`, `scope`, `created_at`, `weight`, `meta`.

### 2.4 Vector index layout

One logical index per layer (L0, L1, L2, L3) so retrieval can target a subset of layers cheaply. Each index stores `{embedding_id, node_id, scope, layer, vector}`. Default model: Voyage-3 (1024d). Mode A fallback: MiniLM-L6-v2 (384d). The model id is recorded on each vector row; a model switch triggers a lazy re-embed pass.

### 2.5 Public schema commitment

The above is published as `schemas/memory-v1.json`. Breaking changes bump the major version and ship a migration script. Frontends and backends MUST validate against the schema.

---

## 3. Compression Pipeline

### 3.1 Triggers

The pipeline is event-driven and asynchronous. Triggers:

1. **Turn count** — every N L0 turns within a session (default N=20).
2. **Idle** — no new turns for T seconds (default T=300).
3. **Session end** — `SessionEnd` hook flushes everything pending.
4. **Manual** — `/mem-compact` slash command or `Memory.compact()` API call.
5. **Scheduled reflection** — a daily cron-like task scans recent L1s for L3 candidates.

Triggers enqueue tasks into `~/.graphmem/queue.db` (SQLite, WAL). Idempotency key = `(scope, session_id, stage, watermark)`.

### 3.2 Stages

Each stage is a separate task type so they can be retried, rate-limited, and observed independently.

**Stage 1 — Episode summarization (L0 → L1)**
- Model: Claude Haiku (Mode B/C); offline heuristic (Mode A).
- Input: a contiguous window of L0 turns within one session, capped at ~6k tokens.
- Output (structured JSON): `{title, summary, key_points, time_range, mentioned_entities: [{name, kind}]}`.
- Side effects: insert L1 node, `DERIVED_FROM` edges to source L0s, `MENTIONS` edges to entities (creating L2 stubs as needed via Stage 2).

**Stage 2 — Entity extraction & merge (L1 → L2)**
- Model: Sonnet (Mode B/C); regex+gazetteer (Mode A).
- For each candidate entity, retrieve top-20 existing L2s in the same scope by name+embedding similarity, then ask the LLM to choose `merge_with: <id>` or `create_new` with reasoning. Update aliases/description on merge.
- Side effects: upsert L2, refresh embedding, add/update `RELATES_TO` / `DEPENDS_ON` edges based on episode evidence.

**Stage 3 — Reflection (L1+ → L3)**
- Model: Sonnet (Mode B/C); skipped (Mode A).
- Triggered only when one of the following is detected:
  - **Contradiction**: two or more L2 entities or L1 episodes assert mutually exclusive facts.
  - **Recurring pattern**: the same entity cluster appears in ≥ K recent L1s AND the cluster changed state or relationship in a non-trivial way.
  - **Explicit user command**: `/mem-reflect` or `Memory.reflect()` called.
- It is NOT triggered on a timer; timer-based scans are expensive and noisy.
- Output: `{insight, kind, evidence_ids, confidence}`.
- Side effects: insert L3 node, `DERIVED_FROM` edges to evidence L1/L2, optional `CONTRADICTS` edge if a previous L3 conflicts.
- **Over-budget skip**: if the daily token budget is exhausted, Stage 3 is skipped first, then Stage 2 sampling is reduced.

### 3.3 Worker contract

- Single `graphmem-worker` process per host by default; lock via `flock` on `~/.graphmem/worker.lock`.
- Concurrency: stage 1 up to 4 in parallel, stages 2/3 serial (to avoid entity merge races).
- Backoff: exponential up to 1h on LLM/network errors; permanent failure after 5 attempts → poison queue + `mem doctor` surfaces it.
- Token accounting: every LLM call records prompt/completion tokens to `metrics.tokens` for budgeting.

### 3.4 GC and lifecycle

- L0 older than `ttl_at` and fully covered by L1 (`DERIVED_FROM` exists) → soft-deleted (flagged `deleted=true`), then hard-deleted after a 7-day grace.
- L1 older than `ttl_at` and superseded by an L3 covering the same time range → soft-deleted with `SUPERSEDES` provenance.
- L2/L3 are never auto-deleted. Manual `mem forget <id>` propagates `SUPERSEDES`/tombstone edges.
- Pinning sets `ttl_at = NULL` and gives the node `score = ∞` floor in retrieval ranking.

---

## 4. Hybrid Retrieval

### 4.1 Public API

```python
def recall(
    self,
    query: str,
    *,
    k: int = 8,
    scope: str | list[str] = "current",
    layers: tuple[Layer, ...] = (L1, L2),
    token_budget: int = 4000,
    time_window: timedelta | None = None,
    edge_types: list[EdgeType] | None = None,
    explain: bool = False,
) -> RecallResult: ...
```

`RecallResult` has `items: list[MemoryItem]`, `formatted: str` (ready to inject into a prompt), `tokens: int`, `latency_ms: int`, and (if `explain=True`) `trace: dict` with the seed list, expansion paths, and per-item scores.

> **`scope="current"` semantics**: When a `Memory` instance is opened with `scope="tw@mac:graphmem"`, passing `"current"` uses that instance default. A list such as `["current", "team:acme:shared"]` merges results from both scopes before ranking. All other values are treated as literal scope strings.

### 4.2 Pipeline (8 steps)

1. **Query rewrite** (optional, B/C only): one cheap Haiku call rewrites the query into 2–3 search variants if `len(query) < 32` or it ends in a question. Cached on `hash(query, scope)`.
2. **Embed query** with the same model used for the targeted layer indices.
3. **Vector top-K seeds** per requested layer; merge with weighted union (default weights: L1=1.0, L2=1.1, L3=1.2, L0=0.6).
4. **Graph expansion**: from each seed, expand up to 2 hops along allowed `edge_types` (default `MENTIONS, RELATES_TO, DEPENDS_ON, SUPERSEDES`), capped at 50 nodes total, deduped by id.
5. **Filter** by scope, `time_window`, and `deleted=false`.
6. **Fusion scoring**:

   ```
   score(n) = w_vec * cos_sim(q, n)
            + w_graph * graph_score(n)
            + w_recent * exp(-age_days / τ)
            + w_layer * layer_prior[n.layer]
            + w_freq * log(1 + access_count)
            + (∞ if n.pinned else 0)
   ```

   Defaults: `w_vec=0.55, w_graph=0.20, w_recent=0.10, w_layer=0.10, w_freq=0.05, τ=14 days`. `graph_score(n)` is the max over expansion paths of `Π edge.weight / hop_count`. All weights are configurable.

7. **MMR diversification** (λ=0.7) among the top 3·k candidates to reduce redundancy, then take top-k.
8. **Format**: render as a markdown block with `[L1] title — summary` / `[L2] name (kind): description` lines, truncating to `token_budget` using a precise tokenizer (tiktoken for OpenAI-compatible, anthropic-tokenizer for Claude).

### 4.3 Caching

- LRU on `hash(query, scope, layers, time_window)` → seeds (60s TTL).
- LRU on `node_id` → expanded neighbors (300s TTL, invalidated on writes touching that node).
- Embedding cache on `hash(text, model)` → vector (24h TTL, persisted in `embeddings_cache.db`).

### 4.4 Token budget contract

- Recall is hard-capped: it returns ≤ `token_budget` tokens or fewer items. The budget is measured, not estimated.
- If the formatted result exceeds budget, items are dropped from the bottom of the ranking until it fits.
- The hook fast path uses a default budget of 1500 tokens to keep injected context lean.

---

## 5. Claude Code Plugin Integration

### 5.1 Repository layout (plugin side)

```
graphmem/                         # repo root
├── packages/graphmem-core/       # the portable Python package
├── plugins/claude-code/
│   ├── plugin.json               # Claude Code plugin manifest
│   ├── hooks/
│   │   ├── on_session_start.sh
│   │   ├── on_user_prompt_submit.sh
│   │   ├── on_post_tool_use.sh
│   │   └── on_session_end.sh
│   ├── mcp_servers/graphmem.json # spawns graphmem-mcp via uvx/pipx
│   ├── commands/                 # 8 slash commands as .md files
│   ├── skills/                   # 3 skills as folders with SKILL.md
│   └── scripts/
│       ├── bootstrap.sh          # installs venv + daemon on first run
│       └── client.py             # talks to daemon over Unix socket
└── docs/
```

### 5.2 Daemon + Unix socket fast path

Hooks are bash scripts that pipe a JSON event into `~/.graphmem/daemon.sock`. The daemon is a long-lived Python process owning the SQLite/Kuzu connections, the embedding model (loaded once), and the LLM clients. Cold start of the bash wrapper is ≈ 3 ms; the daemon warm round-trip is ≈ 20–40 ms. If the socket is missing, the wrapper falls back to `python -m graphmem.cli hook ...` (slower, ≈ 300 ms cold).

**Cold-start budget.** A full Python+Kuzu+embedding-model boot can take 1–3 s. To honor the 50 ms p95 hook target:

- The daemon launches in two phases. **Phase 1** (target < 200 ms) opens the Unix socket and SQLite/Kuzu connections; it can already accept `write_turn` and enqueue tasks. **Phase 2** lazy-loads the embedding model and LLM clients on first recall request.
- If a hook arrives before Phase 2 is ready, `recall()` returns empty (not blocked). The write path still records the turn.
- Idle timeout is **2 hours** (was 30 min in earlier draft). On idle exit, an in-flight queue is checked first.
- The daemon is auto-started by `on_session_start.sh` on first hook of the session, via `nohup` + double-fork.

### 5.3 plugin.json (essentials)

```json
{
  "name": "graphmem",
  "version": "0.1.0",
  "description": "Multi-layer graph memory for Claude Code",
  "hooks": {
    "SessionStart": "hooks/on_session_start.sh",
    "UserPromptSubmit": "hooks/on_user_prompt_submit.sh",
    "PostToolUse": "hooks/on_post_tool_use.sh",
    "SessionEnd": "hooks/on_session_end.sh"
  },
  "mcpServers": { "graphmem": "mcp_servers/graphmem.json" },
  "commands": "commands/",
  "skills": "skills/"
}
```

### 5.4 Hook contracts

- **SessionStart**: ensure daemon, register session id, write a `Session` L0 node with cwd/git ref.
- **UserPromptSubmit**: call `recall()` with `k=8, layers=(L1, L2), token_budget=1500`. Hard timeout **200 ms total**, with two internal stages: vector seed pass (target < 80 ms) → graph expansion (target < 100 ms). On graph-stage timeout, fall back to vector-only seeds and still return a result; only return empty if even the vector seed step misses the budget. Inject the `formatted` string into the user turn via the hook's stdout JSON contract.
- **PostToolUse**: append a tool-call L0 node; enqueue compression check.
- **SessionEnd**: enqueue a final stage-1 task covering the session.

### 5.5 MCP tools (9)

`recall_memory`, `add_memory`, `traverse_graph`, `list_recent_episodes`, `search_entities`, `pin_memory`, `unpin_memory`, `get_node`, `get_stats`. All tools accept `scope` (default current) and return JSON. Tool schemas are published as `schemas/mcp-tools-v1.json` and reused by non-Claude-Code MCP clients.

### 5.6 Slash commands (8)

`/mem-search <query>`, `/mem-graph <node-id>`, `/mem-pin <node-id>`, `/mem-compact`, `/mem-reflect`, `/mem-status`, `/mem-export <path>`, `/mem-doctor`. Each is a thin markdown file that instructs the model to call the corresponding MCP tool.

### 5.7 Skills (3)

- `graphmem:recall-memory` — when the model needs to look something up before answering.
- `graphmem:explore-graph` — when the user asks "why" / "what depends on X".
- `graphmem:pin-insight` — when the model arrives at a durable conclusion worth pinning.

Skills wrap MCP tool calls with usage guidance and few-shot examples.

### 5.8 Bootstrap flow

First `/plugin install graphmem` run:

1. Plugin manifest unpacks into `~/.claude/plugins/.../graphmem/`.
2. On first hook, `bootstrap.sh` runs: creates `~/.graphmem/venv/`, `pip install graphmem-core` from PyPI, writes `~/.graphmem/config.yaml` with detected backends (Mode A by default).
3. Daemon launches; subsequent hooks use the socket.

User can run `mem doctor` to see status, switch modes, set API keys via `mem config set anthropic.api_key ...`.

### 5.9 Runtime directory

```
~/.graphmem/
├── config.yaml
├── venv/
├── db/                      # Kuzu graph + sqlite-vss vectors
├── queue.db                 # SQLite WAL task queue
├── embeddings_cache.db
├── daemon.sock
├── daemon.pid
├── worker.lock
├── logs/{daemon,worker}.log # JSON-lines, rotated daily
└── exports/                 # /mem-export targets
```

### 5.10 Security & privacy

- All data is local in Mode A/B. Nothing leaves the machine except embeddings/LLM calls (Mode B).
- Per-scope opt-out file `.graphmem-ignore` (gitignore syntax) skips paths from compression.
- Secret scrubber runs on every L0 write: regex + entropy heuristic redacts API keys, JWTs, private keys before storage.
- `mem export` and `mem import` produce a signed JSON bundle (`schemas/export-v1.json`); imports require an explicit `--trust` flag.

### 5.11 Failure modes

| Failure | Behavior |
|---------|----------|
| Daemon down | Fallback to direct CLI, log warning |
| LLM rate limited | Queue task with backoff; recall still works on existing data |
| Vector store corrupt | `mem doctor --rebuild` re-embeds from L0/L1/L2 |
| Schema migration | Auto-apply on daemon start with backup to `db.bak.<timestamp>` |
| Disk full | Stop accepting writes, surface red status; never partial-write |

---

## 6. Integration into Other Projects / Agents

### 6.1 Four channels

1. **Python SDK** — `pip install graphmem`; `from graphmem import Memory`.
2. **MCP server** — `uvx graphmem-mcp` or `pipx run graphmem-mcp`; works in any MCP-aware client (Claude Desktop, Cline, Continue, etc.).
3. **HTTP REST** (v2) — `graphmem serve --http :7077`; OpenAPI spec published.
4. **CLI** — `mem write`, `mem recall`, `mem graph`, `mem export` for shell pipelines.

All four share the same `Memory` class internally and the same on-disk store, so a Claude Code session, a LangGraph agent, and a Python script can all read/write the same memory if they point at the same `GRAPHMEM_HOME`.

### 6.2 Adapters (`graphmem.adapters.*`)

Six official adapters in v1:

- `langgraph` — `GraphmemMemory` checkpointer + retrieval node.
- `autogen` — `GraphmemContext` mixin for `ConversableAgent`.
- `crewai` — `GraphmemTool` exposing recall/write to crew agents.
- `dspy` — `dspy.Retrieve`-compatible retriever.
- `openai_agents` — tool functions matching the Agents SDK protocol.
- `anthropic_messages` — middleware that intercepts `messages.create` calls, recalls before, writes after.

Each adapter is ≤ 200 lines and lives under `graphmem.adapters.<framework>`; they are optional installs (`pip install graphmem[langgraph]`).

### 6.3 LangGraph quickstart (illustrative)

```python
from graphmem import Memory
from graphmem.adapters.langgraph import recall_node, write_node

mem = Memory.open(home="~/.graphmem", scope="proj:billing")

graph.add_node("recall", recall_node(mem, k=8))
graph.add_node("agent", my_agent_node)
graph.add_node("write",  write_node(mem))
graph.add_edge("recall", "agent")
graph.add_edge("agent",  "write")
```

### 6.4 Multi-agent sharing

Two agents can share memory by:

- pointing at the same `GRAPHMEM_HOME` (single host), OR
- pointing at the same Mode-C backend via `GRAPHMEM_BACKEND=remote://…`.

`scope` becomes the access boundary. ACLs in Mode C are scope-prefixed.

### 6.5 Cross-machine deployment

Mode C ships a `docker-compose.yaml` with: `graphmem-api` (FastAPI), `graphmem-worker`, Neo4j, Qdrant (or pgvector), Redis (queue), and an optional reverse proxy. Authn: API key per scope; authz: scope-prefix ACL.

### 6.6 Integration checklist (for a new framework)

1. Pick a channel: SDK (in-proc) > MCP (sandboxed) > HTTP (remote) > CLI (script).
2. Set `GRAPHMEM_HOME` (or `GRAPHMEM_BACKEND` for Mode C).
3. Decide `scope` per agent / per project.
4. Wrap or insert `recall()` before LLM calls; call `write_turn()` after.
5. Optionally pin key insights with `pin()`.
6. Run `mem doctor` to confirm.

### 6.7 Comparison with claude-mem

| Dimension | claude-mem | graphmem (target) |
|-----------|------------|-------------------|
| Scope | Personal, Claude Code only | Personal + team, any agent |
| Layers | Summary only | L0/L1/L2/L3 + scope |
| Storage | SQLite + ONNX embeddings | Kuzu + sqlite-vss (A) / Neo4j + Qdrant (C) |
| Retrieval | Vector top-K | Vector + typed graph + fusion |
| Compression | Single-stage | Three-stage (episode/entity/reflection) |
| Distribution | npm/JS plugin | Python plugin + SDK + MCP + HTTP |
| Reflection | No | Yes (L3) |
| Mode A parity | n/a | Required |

---

## 7. Project Structure, Testing, CI, Release

### 7.1 Final repo layout

```
graphmem/
├── packages/
│   └── graphmem-core/
│       ├── pyproject.toml
│       └── src/graphmem/
│           ├── __init__.py        # exports Memory, Layer, EdgeType
│           ├── memory.py          # public Memory class
│           ├── schema/            # pydantic models + JSON schemas
│           ├── stores/
│           │   ├── base.py        # GraphStore, VectorStore ABCs
│           │   ├── kuzu_store.py
│           │   ├── neo4j_store.py
│           │   ├── sqlite_vss_store.py
│           │   ├── qdrant_store.py
│           │   └── pgvector_store.py
│           ├── llm/
│           │   ├── base.py
│           │   ├── anthropic_client.py
│           │   └── noop_client.py
│           ├── embed/
│           │   ├── base.py
│           │   ├── voyage_client.py
│           │   └── sentence_transformer_client.py
│           ├── queue/
│           │   ├── base.py
│           │   ├── sqlite_queue.py
│           │   └── redis_queue.py
│           ├── pipeline/          # compression stages
│           ├── retrieval/         # vector seed, graph expand, fusion, mmr
│           ├── adapters/          # langgraph, autogen, crewai, dspy, …
│           ├── observability/     # logging, metrics, tracing
│           ├── cli.py             # `mem` entry point
│           ├── daemon.py
│           ├── worker.py
│           └── mcp_server.py
├── plugins/claude-code/           # see Section 5
├── deploy/docker/                 # Mode C compose + Dockerfiles
├── schemas/                       # versioned JSON schemas
├── tests/                         # pyramid (Section 7.3)
├── benchmarks/                    # parity vs claude-mem, recall benches
├── docs/
│   ├── superpowers/specs/
│   ├── quickstart-claude-code.md
│   ├── quickstart-langgraph.md
│   ├── quickstart-mcp.md
│   ├── architecture.md
│   └── api-reference.md
└── .github/workflows/
```

### 7.2 Module boundaries (rules)

- `graphmem.memory` is the only module frontends/adapters import.
- `stores`, `llm`, `embed`, `queue` expose only ABCs publicly; concrete classes are picked via factory from `config.yaml`.
- `adapters` imports only from `graphmem` public surface; never reaches into stores or pipeline.
- Cyclic imports are forbidden; enforced by `import-linter` in CI.

### 7.3 Test pyramid (5 layers)

1. **Unit** — pure-python tests on ABC contracts, fusion math, mmr, schema validation. Target ≥ 90% coverage on `memory`, `pipeline`, `retrieval`.
2. **Contract** — every concrete store/llm/embed/queue must pass a shared contract suite (`tests/contracts/`).
3. **Integration** — Memory end-to-end with Mode A backends in a tmpdir; covers compress + recall + GC.
4. **Plugin** — Claude Code hook simulator drives bash hooks against a fake daemon; asserts hook-stdout JSON shape and timing budget.
5. **Parity & bench** — runs claude-mem and graphmem (Mode A) on the same conversations; asserts recall@k ≥ baseline. Also tracks p95 latency.

### 7.4 CI/CD

- GitHub Actions on PR: lint (ruff, black, mypy strict on public api), import-linter, full unit + contract suite, Mode A integration on Linux/macOS.
- Nightly: parity bench, Mode B integration with mocked Anthropic, schema compatibility against last release.
- Release workflow (tag `v*`): build wheels, publish to PyPI, build Claude Code plugin bundle, attach to GitHub release, push docker images to GHCR.

### 7.5 Versioning & compatibility

- SemVer for `graphmem-core` and the plugin (kept in lockstep until v1).
- Public API surface: `Memory`, `Layer`, `EdgeType`, `RecallResult`, MCP tool schemas, JSON schemas. Anything else is internal.
- Schema migrations live in `schemas/migrations/<from>_to_<to>.py`; auto-applied at daemon start with backup.

### 7.6 Release roadmap

- **v0.1 (MVP)**: Mode A end-to-end; Claude Code plugin; SDK; CLI; sqlite-vss + Kuzu; NoOp LLM with heuristic chunker; parity suite green.
- **v0.2**: Mode B (Anthropic Haiku/Sonnet); voyage embeddings; LangGraph + AutoGen adapters; reflection stage.
- **v0.3**: Neo4j store; Qdrant/pgvector stores; OpenAI Agents adapter; Anthropic Messages middleware.
- **v0.4**: MCP server hardening; secret scrubber; export/import; `mem doctor` polish.
- **v1.0**: API freeze, schemas frozen, full docs, signed plugin bundle.
- **v2.0**: HTTP REST API; Mode C docker compose; ACLs; team deployment guide.

### 7.7 Documentation priorities

1. Quickstart × 3 (Claude Code, LangGraph, MCP) — must each work end-to-end in ≤ 20 lines / 5 minutes.
2. Architecture overview (this doc, summarized).
3. API reference (autogen from docstrings).
4. Operations guide (modes, GC, doctor, troubleshooting).
5. Migration & schema versioning.

### 7.8 Success metrics (tracked per release)

- p95 hook overhead, p95 recall latency.
- Recall@k on a fixed benchmark set (≥ baseline claude-mem in Mode A; +20% in Mode B).
- Token cost per session (Mode B), tracked weekly.
- Adapter quickstart success rate (CI smoke).
- Parity-suite pass rate (must be 100%).

### 7.9 Key risks

| Risk | Mitigation |
|------|------------|
| Python cold start in hooks | Daemon + Unix socket; fallback CLI documented as slow path |
| Kuzu maturity | Backend abstraction; Neo4j path tested in CI; data export keeps users portable |
| LLM cost in Mode B | Per-scope budget; Haiku for L1; sampling/skip rules for low-signal sessions |
| Schema drift across frontends | Single JSON schema source; contract tests for every adapter |
| Cross-platform paths | Use `pathlib`, test on Linux/macOS in CI; Windows on best-effort with WSL guidance |
| Concurrent writes corrupting graph | Per-scope SQLite lock + Kuzu transactions; queue serializes stage 2/3 |

---

## 8. Open Questions (to resolve in writing-plans)

1. Should `scope` accept hierarchical paths (`team:acme/billing/api`) for inheritance, or stay flat and lean on `PART_OF`? (Leaning flat for v1.)
2. Default L0 TTL — 7 days vs 14 days? Tied to typical session cadence; revisit after dogfooding.
3. Whether to ship Windows native support in v1 or document WSL only.
4. Whether MCP tool schemas should be auto-generated from pydantic or hand-written (auto-generation is cleaner; verify Claude Code MCP compatibility).
5. Exact daemon Phase-1/Phase-2 lazy-load thresholds (what counts as "ready" for recall vs. write-only) — define in implementation plan.

---

## Appendix A — Public API surface (frozen at v1.0)

- `Memory.open(home: str | Path, scope: str, *, config: dict | None = None) -> Memory`
- `Memory.write_turn(role, content, *, session_id, tool_calls=None, meta=None) -> NodeId`
- `Memory.recall(query, *, k=8, scope="current", layers=(L1, L2), token_budget=4000, time_window=None, edge_types=None, explain=False) -> RecallResult`
- `Memory.graph(node_id, *, depth=2, edge_types=None) -> Subgraph`
- `Memory.pin(node_id) -> None` / `Memory.unpin(node_id) -> None`
- `Memory.compact(*, scope="current") -> CompactReport`
- `Memory.export(path, *, scope=None) -> None` / `Memory.import_(path, *, trust=False) -> None`
- `Memory.stats(*, scope="current") -> Stats`
- Enums: `Layer = {L0, L1, L2, L3}`, `EdgeType` (see 2.3).

## Appendix B — Default config (`~/.graphmem/config.yaml`)

```yaml
mode: A                      # auto-detected on bootstrap
home: ~/.graphmem
scope_default: "${USER}@${HOST}:${PROJECT}"

stores:
  graph: { driver: kuzu, path: db/graph.kuzu }
  vector: { driver: sqlite_vss, path: db/vectors.db }
  queue: { driver: sqlite, path: queue.db }

llm:
  driver: noop                # switch to "anthropic" with api key
  models: { episode: claude-haiku-4, entity: claude-sonnet-4, reflection: claude-sonnet-4 }

embed:
  driver: sentence_transformers
  model: all-MiniLM-L6-v2
  dim: 384

retrieval:
  default_k: 8
  default_token_budget: 4000
  weights: { vec: 0.55, graph: 0.20, recent: 0.10, layer: 0.10, freq: 0.05 }
  layer_prior: { L0: 0.5, L1: 1.0, L2: 1.1, L3: 1.2 }
  expansion: { max_hops: 2, max_nodes: 50 }

compression:
  triggers: { turns: 20, idle_seconds: 300, on_session_end: true }
  reflection: { enabled: true, min_episodes: 5 }
  token_budget:
    daily_limit: 100000         # per-scope, mode B only
    weekly_limit: 500000
    warn_at_percent: 80
    on_exhausted: "skip_reflection"   # or "skip_all_llm" or "raise"

privacy:
  scrub_secrets: true
  ignore_files: [".graphmem-ignore"]
```
