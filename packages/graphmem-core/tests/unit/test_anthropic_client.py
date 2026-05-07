def test_complete_returns_string(monkeypatch):
    from graphmem.llm.anthropic_client import AnthropicLLMClient

    class FakeBlock:
        text = "hello"

    class FakeResponse:
        content = [FakeBlock()]

    client = AnthropicLLMClient(api_key="test", default_model="claude-haiku-4")
    monkeypatch.setattr(
        client.client.messages, "create", lambda **kwargs: FakeResponse()
    )
    result = client.complete("say hi")
    assert isinstance(result, str)
    assert "hello" in result


def test_complete_structured(monkeypatch):
    from graphmem.llm.anthropic_client import AnthropicLLMClient

    class FakeBlock:
        text = '{"title": "Test", "summary": "A test summary", "key_points": ["p1"]}'

    class FakeResponse:
        content = [FakeBlock()]

    client = AnthropicLLMClient(api_key="test", default_model="claude-haiku-4")
    monkeypatch.setattr(
        client.client.messages, "create", lambda **kwargs: FakeResponse()
    )
    result = client.complete_structured("summarize", schema={"type": "object"})
    assert result["title"] == "Test"
    assert result["key_points"] == ["p1"]
