"""Background worker for asynchronous batch compression."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphmem.memory import Memory

from graphmem.schema import CompactReport


class CompressionWorker:
    def __init__(
        self,
        memory: Memory,
        *,
        trigger_turns: int = 10,
        trigger_idle_seconds: float = 300.0,
        poll_interval: float = 5.0,
    ):
        self.memory = memory
        self.trigger_turns = trigger_turns
        self.trigger_idle_seconds = trigger_idle_seconds
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10.0)

    def _loop(self) -> None:
        while self._running:
            try:
                self.run_once()
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def run_once(self) -> CompactReport:
        with self._lock:
            return self._process_queue()

    def _process_queue(self) -> CompactReport:
        queue = self.memory.queue
        groups = queue.dequeue_by_session(batch_size=20)
        if not groups:
            return CompactReport(episodes_created=0)

        total_episodes = 0
        for session_id, tasks in groups.items():
            if len(tasks) < 2:
                for t in tasks:
                    queue.complete(t["id"])
                continue

            # Build actual turn objects from stored node IDs
            turn_nodes = []
            for t in tasks:
                node_id = t["payload"].get("node_id")
                if node_id:
                    node = self.memory.graph_store.get_node(node_id)
                    if node is not None:
                        turn_nodes.append(node)

            if len(turn_nodes) < 2:
                for t in tasks:
                    queue.complete(t["id"])
                continue

            # Compress this session
            try:
                self.memory._compress_session(session_id)
                total_episodes += 1
            except Exception:
                for t in tasks:
                    queue.fail(t["id"], "compression failed")
                continue

            for t in tasks:
                queue.complete(t["id"])

        return CompactReport(episodes_created=total_episodes)
