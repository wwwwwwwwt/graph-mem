"""Configuration loading and defaults."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    driver: str = "noop"
    api_key: str | None = None
    models: dict[str, str] = Field(default_factory=dict)


class EmbedConfig(BaseModel):
    driver: str = "sentence_transformers"
    model: str = "all-MiniLM-L6-v2"
    dim: int = 384


class RetrievalConfig(BaseModel):
    default_k: int = 8
    default_token_budget: int = 4000
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "vec": 0.55,
            "graph": 0.20,
            "recent": 0.10,
            "layer": 0.10,
            "freq": 0.05,
        }
    )
    layer_prior: dict[str, float] = Field(
        default_factory=lambda: {
            "L0": 0.5,
            "L1": 1.0,
            "L2": 1.1,
            "L3": 1.2,
        }
    )


class CompressionConfig(BaseModel):
    triggers: dict[str, int | bool] = Field(
        default_factory=lambda: {
            "turns": 20,
            "idle_seconds": 300,
            "on_session_end": True,
        }
    )


class Config(BaseModel):
    mode: str = "A"
    home: str = "~/.graphmem"
    scope_default: str = "${USER}@${HOST}:${PROJECT}"
    stores: dict[str, dict] = Field(
        default_factory=lambda: {
            "graph": {"driver": "kuzu", "path": "db/graph.kuzu"},
            "vector": {"driver": "numpy", "path": "db/vectors", "dim": 384},
            "queue": {"driver": "sqlite", "path": "queue.db"},
        }
    )
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)


def load_config(path: str | None = None) -> Config:
    if path is None:
        path = str(Path.home() / ".graphmem" / "config.yaml")
    p = Path(path)
    if p.exists():
        with open(p, "r") as f:
            data = yaml.safe_load(f)
        return Config(**data)
    return Config()
