import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig


def init_run_dir(config: RunConfig) -> Path:
    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)

    # Save config copy
    config_dst = run_dir / "config.yaml"
    if not config_dst.exists():
        import yaml

        with open(config_dst, "w") as f:
            yaml.dump(
                {
                    "run_name": config.run_name,
                    "model": {
                        "provider": config.model.provider,
                        "model_name": config.model.model_name,
                        "temperature": config.model.temperature,
                        "max_tokens": config.model.max_tokens,
                    },
                    "judge": {
                        "provider": config.judge.provider,
                        "model_name": config.judge.model_name,
                        "temperature": config.judge.temperature,
                        "max_tokens": config.judge.max_tokens,
                    },
                },
                f,
                default_flow_style=False,
            )

    return run_dir


def write_sample_result(run_dir: Path, result: dict) -> None:
    task_name = result["task_name"]
    task_dir = run_dir / "samples" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    path = task_dir / f"{result['sample_id']}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def load_completed_ids(run_dir: Path) -> set[str]:
    completed = set()
    samples_dir = run_dir / "samples"
    if not samples_dir.exists():
        return completed

    for task_dir in samples_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for result_file in task_dir.glob("*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                if data.get("error") is None:
                    completed.add(data["sample_id"])
            except (json.JSONDecodeError, KeyError):
                pass

    return completed


def write_summary(run_dir: Path, all_results: list[dict], config: RunConfig) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    by_turn: dict[int, list[dict]] = defaultdict(list)

    total_model_input = 0
    total_model_output = 0
    total_judge_input = 0
    total_judge_output = 0
    errors = 0

    for r in all_results:
        if r.get("error"):
            errors += 1
            continue

        task = r["task_name"]
        turn = r["turn"]
        by_task[task].append(r)
        by_turn[turn].append(r)

        mu = r.get("model_usage", {})
        ju = r.get("judge_usage", {})
        total_model_input += mu.get("input_tokens", 0)
        total_model_output += mu.get("output_tokens", 0)
        total_judge_input += ju.get("input_tokens", 0)
        total_judge_output += ju.get("output_tokens", 0)

    def _agg(results: list[dict]) -> dict:
        verdicts = [r["judge_verdict"] for r in results]
        scores = [r["judge_score"] for r in results]
        return {
            "total": len(results),
            "equivalent": verdicts.count("equivalent"),
            "partially_equivalent": verdicts.count("partially_equivalent"),
            "not_equivalent": verdicts.count("not_equivalent"),
            "mean_score": round(sum(scores) / len(scores), 3) if scores else 0,
        }

    scored = [r for r in all_results if not r.get("error")]

    summary = {
        "run_name": config.run_name,
        "model": config.model.model_name,
        "judge": config.judge.model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(all_results),
        "overall": _agg(scored),
        "errors": errors,
        "by_task": {task: _agg(results) for task, results in sorted(by_task.items())},
        "by_turn": {
            str(turn): _agg(results)
            for turn, results in sorted(by_turn.items())
        },
        "token_usage": {
            "model_input_tokens": total_model_input,
            "model_output_tokens": total_model_output,
            "judge_input_tokens": total_judge_input,
            "judge_output_tokens": total_judge_output,
        },
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
