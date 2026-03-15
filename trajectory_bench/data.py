import json
from dataclasses import dataclass
from pathlib import Path

from .config import RunConfig


@dataclass
class Round:
    round_id: str
    prompt: list[dict]
    completion: str
    metadata: dict


def _make_round_id(meta: dict) -> str:
    return f"{meta['task_name']}__{meta['trial_id']}__turn{meta['turn']}"


def load_rounds(config: RunConfig) -> list[Round]:
    path = Path(config.data_path)
    rounds: list[Round] = []

    with open(path) as f:
        for line in f:
            raw = json.loads(line)
            meta = raw["metadata"]
            r = Round(
                round_id=_make_round_id(meta),
                prompt=raw["prompt"],
                completion=raw["completion"],
                metadata=meta,
            )

            if config.filter_tasks and meta["task_name"] not in config.filter_tasks:
                continue
            if config.filter_turns is not None and meta["turn"] not in config.filter_turns:
                continue

            rounds.append(r)

    if config.max_rounds is not None:
        rounds = rounds[: config.max_rounds]

    return rounds


def estimate_tokens(rounds: list[Round]) -> dict:
    """Rough token estimate (1 token ~ 4 chars)."""
    total_prompt_chars = sum(
        sum(len(m["content"]) for m in r.prompt) for r in rounds
    )
    total_completion_chars = sum(len(r.completion) for r in rounds)

    return {
        "n_rounds": len(rounds),
        "est_model_input_tokens": total_prompt_chars // 4,
        "est_model_output_tokens": total_completion_chars // 4,
        "est_judge_input_tokens": (total_completion_chars * 2 + len(rounds) * 2000) // 4,
        "est_judge_output_tokens": len(rounds) * 100,
    }
