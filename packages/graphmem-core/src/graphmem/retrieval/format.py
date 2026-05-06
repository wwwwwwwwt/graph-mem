"""Result formatting for recall."""

from graphmem.schema import MemoryItem, Layer


def format_results(items: list[MemoryItem], token_budget: int = 4000) -> str:
    lines = []
    tokens_used = 0
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
