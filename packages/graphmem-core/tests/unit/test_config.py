from graphmem.config import Config, load_config


def test_default_config():
    cfg = Config()
    assert cfg.stores["graph"]["driver"] == "kuzu"
    assert cfg.embed.driver == "sentence_transformers"
    assert cfg.llm.driver == "noop"


def test_load_config_from_file(tmp_home):
    config_path = tmp_home / "config.yaml"
    config_path.write_text("mode: B\nllm:\n  driver: anthropic\n")
    cfg = load_config(str(config_path))
    assert cfg.mode == "B"
    assert cfg.llm.driver == "anthropic"
