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

    def write_after_reply(
        self, messages: list[dict], session_id: str = "default"
    ) -> None:
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                self._graphmem.write_turn(role, content, session_id=session_id)
