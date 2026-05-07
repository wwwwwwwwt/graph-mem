"""Public Memory API."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ulid import ULID

from graphmem.config import Config, load_config
from graphmem.embed.sentence_transformer import SentenceTransformerEmbedClient
from graphmem.llm.noop import NoOpLLMClient
from graphmem.pipeline.episode import HeuristicEpisodeSummarizer, LLMEpisodeSummarizer
from graphmem.pipeline.entity import EntityExtractor
from graphmem.pipeline.reflection import ReflectionGenerator
from graphmem.queue.sqlite_queue import SQLiteTaskQueue
from graphmem.retrieval.format import format_results
from graphmem.retrieval.search import SearchEngine
from graphmem.schema import (
    CompactReport,
    EdgeType,
    L0Turn,
    L1Episode,
    L2Entity,
    L3Reflection,
    Layer,
    MemoryEdge,
    MemoryNode,
    RecallResult,
    Stats,
    Subgraph,
)
from graphmem.stores.kuzu_store import KuzuGraphStore
from graphmem.stores.numpy_store import NumpyVectorStore


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
        if isinstance(self.llm_client, NoOpLLMClient):
            self.summarizer = HeuristicEpisodeSummarizer(
                trigger_turns=config.compression.triggers.get("turns", 20),
                trigger_idle_seconds=config.compression.triggers.get("idle_seconds", 300),
            )
        else:
            self.summarizer = LLMEpisodeSummarizer(
                llm_client=self.llm_client,
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

        # LLM backend selection
        llm_cfg = cfg.llm
        if llm_cfg.driver in ("anthropic",) and llm_cfg.api_key:
            from graphmem.llm.anthropic_client import AnthropicLLMClient
            llm_client = AnthropicLLMClient(
                api_key=llm_cfg.api_key,
                default_model=llm_cfg.models.get("episode", llm_cfg.default_model or "claude-haiku-4"),
            )
        elif llm_cfg.driver in ("openai_compatible", "openai", "deepseek") and llm_cfg.api_key:
            from graphmem.llm.openai_compatible import OpenAILLMClient
            llm_client = OpenAILLMClient(
                api_key=llm_cfg.api_key,
                base_url=llm_cfg.base_url,
                default_model=llm_cfg.default_model or llm_cfg.models.get("episode", "gpt-4o-mini"),
                extra_params=getattr(llm_cfg, "extra_params", None) or {},
            )
        else:
            llm_client = NoOpLLMClient()

        # Embed backend selection
        embed_cfg = cfg.embed
        if embed_cfg.driver == "voyage" and getattr(embed_cfg, "api_key", None):
            from graphmem.embed.voyage_client import VoyageEmbedClient
            embed_client = VoyageEmbedClient(
                api_key=embed_cfg.api_key, model=embed_cfg.model
            )
        elif embed_cfg.driver == "bge":
            from graphmem.embed.bge_client import BGEEmbedClient
            embed_client = BGEEmbedClient(
                model_name=embed_cfg.model,
                dim=embed_cfg.dim,
            )
        else:
            embed_client = SentenceTransformerEmbedClient(model_name=embed_cfg.model)

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

        embeddings = self.embed_client.embed([content])
        self.vector_store.insert(
            embedding_id=f"emb-{node_id}",
            node_id=node_id,
            scope=self.scope,
            layer=Layer.L0.value,
            vector=embeddings[0],
        )

        # Enqueue compression task for async batch processing
        self.queue.enqueue("compress", {
            "session_id": session_id,
            "node_id": node_id,
            "scope": self.scope,
        })

        return node_id

    def _count_turns_in_session(self, session_id: str) -> int:
        nodes = self.graph_store.query_nodes(scope=self.scope, layer="L0")
        return sum(1 for n in nodes if getattr(n, "session_id", "") == session_id)

    def _compress_session(self, session_id: str) -> None:
        nodes = self.graph_store.query_nodes(scope=self.scope, layer="L0")
        turns = [n for n in nodes if getattr(n, "session_id", "") == session_id]
        turns.sort(key=lambda t: t.turn_index)

        # Skip turns already linked to an L1 episode to avoid duplicate compression
        uncompressed = []
        for turn in turns:
            neighbors = self.graph_store.get_neighbors(
                turn.id, direction="in", edge_types=[EdgeType.DERIVED_FROM]
            )
            if not neighbors:
                uncompressed.append(turn)
        turns = uncompressed

        if len(turns) < 2:
            return

        episode = self.summarizer.summarize(turns)
        episode.id = f"L1-{ULID()}"
        self.graph_store.create_node(episode)

        for turn in turns:
            edge = MemoryEdge(
                id=f"edge-{episode.id}-{turn.id}",
                type=EdgeType.DERIVED_FROM,
                from_id=episode.id,
                to_id=turn.id,
                scope=self.scope,
            )
            self.graph_store.create_edge(edge)

        text = f"{episode.title}\n{episode.summary}"
        embeddings = self.embed_client.embed([text])
        self.vector_store.insert(
            embedding_id=f"emb-{episode.id}",
            node_id=episode.id,
            scope=self.scope,
            layer=Layer.L1.value,
            vector=embeddings[0],
        )

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
            generator = ReflectionGenerator(
                self.llm_client,
                min_episodes=self.config.compression.triggers.get("min_episodes", 5),
            )
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
        start = datetime.now(timezone.utc)
        effective_scope = self.scope if scope == "current" else scope
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
        latency = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)

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
        return Stats(
            scope=effective_scope,
            nodes_by_layer=counts,
        )

    def close(self) -> None:
        self.graph_store.close()
        self.vector_store.close()
        self.queue.close()
