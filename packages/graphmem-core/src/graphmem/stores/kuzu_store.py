"""Kuzu-based implementation of GraphStore."""

import json
from datetime import datetime, timezone
from typing import Any

import kuzu

from graphmem.schema import EdgeType, Layer, L0Turn, L1Episode, L2Entity, L3Reflection, MemoryEdge, MemoryNode
from graphmem.stores.base import GraphStore


def _dt_to_str(dt: datetime | None) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _str_to_dt(s: str) -> datetime | None:
    if not s:
        return None
    # Kuzu may return with or without Z
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _node_to_dict(node: MemoryNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "layer": node.layer.value,
        "scope": node.scope,
        "created_at": _dt_to_str(node.created_at),
        "updated_at": _dt_to_str(node.updated_at),
        "source": node.source,
        "tokens": node.tokens,
        "pinned": node.pinned,
        "ttl_at": _dt_to_str(node.ttl_at),
        "meta": json.dumps(node.meta),
        "content": node.content,
    }


def _row_to_node(row: dict[str, Any]) -> MemoryNode:
    layer = Layer(row["layer"])
    meta = json.loads(row.get("meta", "{}"))
    data = {
        "id": row["id"],
        "layer": layer,
        "scope": row["scope"],
        "created_at": _str_to_dt(row.get("created_at", "")),
        "updated_at": _str_to_dt(row.get("updated_at", "")),
        "source": row.get("source", ""),
        "tokens": row.get("tokens", 0),
        "pinned": row.get("pinned", False),
        "ttl_at": _str_to_dt(row.get("ttl_at", "")),
        "meta": meta,
    }
    match layer:
        case Layer.L0:
            return L0Turn(
                **data,
                role=row.get("role", ""),
                content=row.get("content", ""),
                tool_calls=json.loads(row.get("tool_calls", "null")),
                session_id=row.get("session_id", ""),
                turn_index=row.get("turn_index", 0),
            )
        case Layer.L1:
            return L1Episode(
                **data,
                title=row.get("title", ""),
                summary=row.get("summary", ""),
                key_points=json.loads(row.get("key_points", "[]")),
                participants=json.loads(row.get("participants", "[]")),
                time_range=None,
                content=row.get("content", ""),
            )
        case Layer.L2:
            return L2Entity(
                **data,
                name=row.get("name", ""),
                kind=row.get("kind", ""),
                aliases=json.loads(row.get("aliases", "[]")),
                description=row.get("description", ""),
                confidence=row.get("confidence", 1.0),
                content=row.get("content", ""),
            )
        case Layer.L3:
            return L3Reflection(
                **data,
                insight=row.get("insight", ""),
                evidence_ids=json.loads(row.get("evidence_ids", "[]")),
                confidence=row.get("confidence", 1.0),
                kind=row.get("kind_ref", ""),
                content=row.get("content", ""),
            )
        case _:
            return MemoryNode(**data, content=row.get("content", ""))


class KuzuGraphStore(GraphStore):
    def __init__(self, db_path: str):
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._init_schema()

    def _init_schema(self) -> None:
        try:
            self.conn.execute(
                "CREATE NODE TABLE MemoryNode("
                "id STRING, layer STRING, scope STRING, created_at STRING, "
                "updated_at STRING, source STRING, tokens INT64, pinned BOOL, "
                "ttl_at STRING, meta STRING, content STRING, "
                "role STRING, tool_calls STRING, session_id STRING, turn_index INT64, "
                "title STRING, summary STRING, key_points STRING, participants STRING, "
                "name STRING, kind STRING, aliases STRING, description STRING, confidence DOUBLE, "
                "insight STRING, evidence_ids STRING, kind_ref STRING, "
                "PRIMARY KEY(id))"
            )
        except Exception:
            pass  # table already exists
        try:
            self.conn.execute(
                "CREATE REL TABLE MemoryEdge("
                "FROM MemoryNode TO MemoryNode, "
                "type STRING, scope STRING, created_at STRING, "
                "weight DOUBLE, meta STRING, MANY_MANY)"
            )
        except Exception:
            pass  # table already exists

    def _escape(self, v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int) or isinstance(v, float):
            return str(v)
        return "'" + str(v).replace("'", "\\'") + "'"

    def create_node(self, node: MemoryNode) -> None:
        d = _node_to_dict(node)
        pairs = []
        for k, v in d.items():
            pairs.append(f"{k}: {self._escape(v)}")
        # layer-specific fields
        if isinstance(node, L0Turn):
            pairs.extend([
                f"role: {self._escape(node.role)}",
                f"tool_calls: {self._escape(json.dumps(node.tool_calls or []))}",
                f"session_id: {self._escape(node.session_id)}",
                f"turn_index: {self._escape(node.turn_index)}",
            ])
        elif isinstance(node, L1Episode):
            pairs.extend([
                f"title: {self._escape(node.title)}",
                f"summary: {self._escape(node.summary)}",
                f"key_points: {self._escape(json.dumps(node.key_points))}",
                f"participants: {self._escape(json.dumps(node.participants))}",
            ])
        elif isinstance(node, L2Entity):
            pairs.extend([
                f"name: {self._escape(node.name)}",
                f"kind: {self._escape(node.kind)}",
                f"aliases: {self._escape(json.dumps(node.aliases))}",
                f"description: {self._escape(node.description)}",
                f"confidence: {self._escape(node.confidence)}",
            ])
        elif isinstance(node, L3Reflection):
            pairs.extend([
                f"insight: {self._escape(node.insight)}",
                f"evidence_ids: {self._escape(json.dumps(node.evidence_ids))}",
                f"confidence: {self._escape(node.confidence)}",
                f"kind_ref: {self._escape(node.kind)}",
            ])

        cypher = f"CREATE (n:MemoryNode {{{', '.join(pairs)}}})"
        self.conn.execute(cypher)

    def get_node(self, node_id: str) -> MemoryNode | None:
        result = self.conn.execute(f"MATCH (n:MemoryNode) WHERE n.id = '{node_id}' RETURN n")
        if result.has_next():
            row = result.get_next()
            node_dict = row[0]
            return _row_to_node(node_dict)
        return None

    def update_node(self, node: MemoryNode) -> None:
        # Kuzu update is limited; delete and recreate
        self.conn.execute(f"MATCH (n:MemoryNode) WHERE n.id = '{node.id}' DELETE n")
        self.create_node(node)

    def create_edge(self, edge: MemoryEdge) -> None:
        cypher = (
            f"MATCH (a:MemoryNode), (b:MemoryNode) "
            f"WHERE a.id = '{edge.from_id}' AND b.id = '{edge.to_id}' "
            f"CREATE (a)-[:MemoryEdge "
            f"{{type: '{edge.type.value}', scope: '{edge.scope}', "
            f"created_at: '{_dt_to_str(edge.created_at)}', weight: {edge.weight}, "
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
            edge_data = row[0]
            node_data = row[1]
            edge_type = EdgeType(edge_data["type"])
            if edge_types and edge_type not in edge_types:
                continue
            edge = MemoryEdge(
                id=f"{edge_data['_src']['offset']}-{edge_data['_dst']['offset']}",
                type=edge_type,
                from_id=node_data["id"] if direction == "in" else node_id,
                to_id=node_id if direction == "in" else node_data["id"],
                scope=edge_data["scope"],
                weight=edge_data.get("weight", 1.0),
                meta=json.loads(edge_data.get("meta", "{}")),
                created_at=_str_to_dt(edge_data.get("created_at", "")),
            )
            if direction == "in":
                edge.from_id, edge.to_id = edge.to_id, edge.from_id
            neighbors.append((edge, _row_to_node(node_data)))
        return neighbors

    def query_nodes(
        self,
        *,
        scope: str | None = None,
        layer: str | None = None,
        limit: int = 100,
    ) -> list[MemoryNode]:
        conditions = []
        if scope:
            conditions.append(f"n.scope = '{scope}'")
        if layer:
            conditions.append(f"n.layer = '{layer}'")
        where_str = " AND ".join(conditions) if conditions else "1=1"
        cypher = f"MATCH (n:MemoryNode) WHERE {where_str} RETURN n LIMIT {limit}"
        result = self.conn.execute(cypher)
        nodes = []
        while result.has_next():
            row = result.get_next()
            nodes.append(_row_to_node(row[0]))
        return nodes

    def count_nodes(self, scope: str | None = None) -> dict[str, int]:
        where = f"WHERE n.scope = '{scope}'" if scope else ""
        cypher = f"MATCH (n:MemoryNode) {where} RETURN n.layer, COUNT(n)"
        result = self.conn.execute(cypher)
        counts: dict[str, int] = {}
        while result.has_next():
            row = result.get_next()
            counts[row[0]] = row[1]
        return counts

    def close(self) -> None:
        self.conn.close()
