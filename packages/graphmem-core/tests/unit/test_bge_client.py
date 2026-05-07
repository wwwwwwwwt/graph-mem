"""Test BGE embed client."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphmem.embed.bge_client import BGEEmbedClient


def test_bge_embeds_and_normalizes():
    client = BGEEmbedClient(model_name="all-MiniLM-L6-v2", dim=384)
    vectors = client.embed(["hello world", "goodbye"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384

    # Normalized vectors should have length ~1.0
    import math
    norm = math.sqrt(sum(x * x for x in vectors[0]))
    assert abs(norm - 1.0) < 1e-5
