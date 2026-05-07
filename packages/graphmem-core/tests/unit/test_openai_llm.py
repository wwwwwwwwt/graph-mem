def test_complete_returns_string(monkeypatch):
    from graphmem.llm.openai_compatible import OpenAILLMClient

    class FakeMessage:
        content = "hello"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    client = OpenAILLMClient(api_key="test", default_model="deepseek-chat")
    monkeypatch.setattr(
        client.client.chat.completions, "create", lambda **kwargs: FakeResponse()
    )
    result = client.complete("say hi")
    assert isinstance(result, str)
    assert "hello" in result


def test_complete_structured(monkeypatch):
    from graphmem.llm.openai_compatible import OpenAILLMClient

    class FakeMessage:
        content = '{"title": "Test", "summary": "A test summary", "key_points": ["p1"]}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    client = OpenAILLMClient(api_key="test", default_model="deepseek-chat")
    monkeypatch.setattr(
        client.client.chat.completions, "create", lambda **kwargs: FakeResponse()
    )
    result = client.complete_structured("summarize", schema={"type": "object"})
    assert result["title"] == "Test"
    assert result["key_points"] == ["p1"]


def test_deepseek_mode_detection(tmp_home):
    from graphmem.memory import Memory
    from graphmem.llm.openai_compatible import OpenAILLMClient

    cfg = {
        "mode": "B",
        "llm": {
            "driver": "deepseek",
            "provider": "deepseek",
            "api_key": "test-key",
            "base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek-chat",
        },
    }
    mem = Memory.open(str(tmp_home), scope="s1", config=cfg)
    assert isinstance(mem.llm_client, OpenAILLMClient)
    mem.close()
