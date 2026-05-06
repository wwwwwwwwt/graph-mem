"""Numpy + SQLite vector store implementation."""

import sqlite3
from pathlib import Path

import numpy as np

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

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "row_idx INTEGER PRIMARY KEY, "
                "embedding_id TEXT UNIQUE, "
                "node_id TEXT, "
                "scope TEXT, "
                "layer TEXT, "
                "model_id TEXT DEFAULT '')"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scope ON embeddings(scope)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_layer ON embeddings(layer)")
            conn.commit()

    def _load_vectors(self) -> None:
        if self.npz_path.exists():
            data = np.load(str(self.npz_path))
            self._vectors = data["vectors"].astype(np.float32)
        else:
            self._vectors = np.zeros((0, self.dim), dtype=np.float32)

    def _save_vectors(self) -> None:
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

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO embeddings (row_idx, embedding_id, node_id, scope, layer) "
                "VALUES (?, ?, ?, ?, ?)",
                (row_idx, embedding_id, node_id, scope, layer),
            )
            conn.commit()
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
        norms = np.linalg.norm(self._vectors, axis=1)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        sims = np.dot(self._vectors, q) / (norms * q_norm + 1e-10)

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            where = []
            params: list = []
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

        candidates = []
        for idx in range(len(sims)):
            if idx in self._deleted or idx not in valid_rows:
                continue
            candidates.append((valid_rows[idx], float(sims[idx])))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def delete(self, embedding_id: str) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT row_idx FROM embeddings WHERE embedding_id = ?", (embedding_id,)
            )
            row = cursor.fetchone()
            if row:
                self._deleted.add(row[0])

    def close(self) -> None:
        self._save_vectors()
