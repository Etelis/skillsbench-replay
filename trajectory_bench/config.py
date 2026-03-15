from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    provider: str = "openai"  # "openai" (vLLM, TGI, etc.) or "anthropic"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 8192
    base_url: str | None = "http://localhost:8000/v1"  # vLLM endpoint
    api_key: str | None = None  # Override API key (defaults to env var)


@dataclass
class JudgeConfig:
    provider: str = "openai"  # "openai" (vLLM, TGI, etc.) or "anthropic"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 2048
    base_url: str | None = "http://localhost:8000/v1"
    api_key: str | None = None


@dataclass
class RunConfig:
    run_name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    data_path: str = "data/samples/all-solved.jsonl"
    filter_tasks: list[str] | None = None
    filter_turns: list[int] | None = None
    max_samples: int | None = None
    max_concurrent: int = 5
    output_dir: str = "results"
    max_retries: int = 2


def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    model_cfg = ModelConfig(**raw.pop("model", {}))
    judge_cfg = JudgeConfig(**raw.pop("judge", {}))
    return RunConfig(model=model_cfg, judge=judge_cfg, **raw)
