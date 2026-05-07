"""SQLite-based task queue."""

import json
import sqlite3
from datetime import datetime, timezone

from graphmem.queue.base import TaskQueue


class SQLiteTaskQueue(TaskQueue):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
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

    def enqueue(self, task_type: str, payload: dict) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO tasks (task_type, payload, created_at) VALUES (?, ?, ?)",
                (task_type, json.dumps(payload), datetime.now(timezone.utc).isoformat()),
            )
            task_id = str(cursor.lastrowid)
            conn.commit()
        return task_id

    def dequeue(self, *, limit: int = 1) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
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
                    (datetime.now(timezone.utc).isoformat(), task_id),
                )
                tasks.append({
                    "id": str(task_id),
                    "task_type": task_type,
                    "payload": json.loads(payload),
                })
            conn.commit()
        return tasks

    def complete(self, task_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE tasks SET status = 'completed', completed_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), int(task_id)),
            )
            conn.commit()

    def fail(self, task_id: str, error: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE tasks SET status = 'failed', error = ? WHERE id = ?",
                (error, int(task_id)),
            )
            conn.commit()

    def dequeue_by_session(self, *, batch_size: int = 20) -> dict[str, list[dict]]:
        """Group pending tasks by session_id for batch compression."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, task_type, payload FROM tasks WHERE status = 'pending' "
                "ORDER BY created_at LIMIT ?",
                (batch_size,),
            )
            rows = cursor.fetchall()
            if not rows:
                return {}

            tasks = []
            for row in rows:
                task_id, task_type, payload = row
                cursor.execute(
                    "UPDATE tasks SET status = 'running', started_at = ? WHERE id = ?",
                    (datetime.now(timezone.utc).isoformat(), task_id),
                )
                tasks.append({
                    "id": str(task_id),
                    "task_type": task_type,
                    "payload": json.loads(payload),
                })
            conn.commit()

        groups: dict[str, list[dict]] = {}
        for t in tasks:
            sid = t["payload"].get("session_id", "_no_session")
            groups.setdefault(sid, []).append(t)
        return groups

    def count_pending(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'")
            return cursor.fetchone()[0]

    def close(self) -> None:
        pass
