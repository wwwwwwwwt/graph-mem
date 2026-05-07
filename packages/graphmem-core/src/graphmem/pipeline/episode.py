"""Heuristic episode summarizer for Mode A."""

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

        title = "Untitled"
        for t in turns:
            if t.role == "user":
                title = t.content.strip().split("\n")[0][:80]
                break

        messages = [f"{t.role}: {t.content}" for t in turns]
        summary = "\n".join(messages)
        if len(summary) > 1000:
            summary = summary[:997] + "..."

        text = " ".join(t.content for t in turns)
        key_points = self._extract_key_phrases(text)
        participants = list({t.role for t in turns})
        time_range = (turns[0].created_at, turns[-1].created_at)

        return L1Episode(
            id="",
            scope=turns[0].scope,
            layer=Layer.L1,
            title=title,
            summary=summary,
            time_range=time_range,
            key_points=key_points[:5],
            participants=participants,
        )

    def _extract_key_phrases(self, text: str) -> list[str]:
        phrases = set()
        for match in re.finditer(r'"([^"]{3,60})"', text):
            phrases.add(match.group(1))
        for match in re.finditer(r"[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*", text):
            phrase = match.group(0)
            if len(phrase) <= 60:
                phrases.add(phrase)
        return sorted(phrases)[:10]


class LLMEpisodeSummarizer:
    def __init__(self, llm_client, trigger_turns: int = 20, trigger_idle_seconds: int = 300):
        self.llm_client = llm_client
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
                "mentioned_entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "kind": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["title", "summary", "key_points"],
        }
        try:
            result = self.llm_client.complete_structured(prompt, schema=schema, max_tokens=1024)
        except Exception:
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
