import json
from dataclasses import dataclass
from pathlib import Path

from .config import RunConfig


@dataclass
class Sample:
    sample_id: str
    prompt: list[dict]
    completion: str
    metadata: dict


def _make_sample_id(meta: dict) -> str:
    return f"{meta['task_name']}__{meta['trial_id']}__turn{meta['turn']}"


def load_samples(config: RunConfig) -> list[Sample]:
    path = Path(config.data_path)
    samples: list[Sample] = []

    with open(path) as f:
        for line in f:
            raw = json.loads(line)
            meta = raw["metadata"]
            sample = Sample(
                sample_id=_make_sample_id(meta),
                prompt=raw["prompt"],
                completion=raw["completion"],
                metadata=meta,
            )

            if config.filter_tasks and meta["task_name"] not in config.filter_tasks:
                continue
            if config.filter_turns is not None and meta["turn"] not in config.filter_turns:
                continue

            samples.append(sample)

    if config.max_samples is not None:
        samples = samples[: config.max_samples]

    return samples


def estimate_tokens(samples: list[Sample]) -> dict:
    """Rough token estimate (1 token ~ 4 chars)."""
    total_prompt_chars = sum(
        sum(len(m["content"]) for m in s.prompt) for s in samples
    )
    total_completion_chars = sum(len(s.completion) for s in samples)

    return {
        "n_samples": len(samples),
        "est_model_input_tokens": total_prompt_chars // 4,
        "est_model_output_tokens": total_completion_chars // 4,
        "est_judge_input_tokens": (total_completion_chars * 2 + len(samples) * 2000) // 4,
        "est_judge_output_tokens": len(samples) * 100,
    }
