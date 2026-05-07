from graphmem.memory import Memory
from graphmem.llm.noop import NoOpLLMClient


def test_mode_a_uses_noop_llm(tmp_home):
    mem = Memory.open(str(tmp_home), scope="s1")
    assert isinstance(mem.llm_client, NoOpLLMClient)
    mem.close()


def test_mode_b_uses_anthropic_llm(tmp_home, monkeypatch):
    from graphmem.llm.anthropic_client import AnthropicLLMClient

    cfg = {
        "mode": "B",
        "llm": {
            "driver": "anthropic",
            "api_key": "test-key",
            "models": {"episode": "claude-haiku-4"},
        },
    }
    mem = Memory.open(str(tmp_home), scope="s1", config=cfg)
    assert isinstance(mem.llm_client, AnthropicLLMClient)
    mem.close()
