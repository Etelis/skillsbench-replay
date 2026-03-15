import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig


def init_run_dir(config: RunConfig) -> Path:
    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rounds").mkdir(exist_ok=True)

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
                        "base_url": config.model.base_url,
                        "temperature": config.model.temperature,
                        "max_tokens": config.model.max_tokens,
                    },
                    "judge": {
                        "provider": config.judge.provider,
                        "model_name": config.judge.model_name,
                        "base_url": config.judge.base_url,
                        "temperature": config.judge.temperature,
                        "max_tokens": config.judge.max_tokens,
                    },
                },
                f,
                default_flow_style=False,
            )

    return run_dir


def write_round_result(run_dir: Path, result: dict) -> None:
    task_name = result["task_name"]
    task_dir = run_dir / "rounds" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    path = task_dir / f"{result['round_id']}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def load_completed_ids(run_dir: Path) -> set[str]:
    completed = set()
    rounds_dir = run_dir / "rounds"
    if not rounds_dir.exists():
        return completed

    for task_dir in rounds_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for result_file in task_dir.glob("*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                if data.get("error") is None:
                    completed.add(data["round_id"])
            except (json.JSONDecodeError, KeyError):
                pass

    return completed


def _criterion_stats(results: list[dict], criterion: str) -> dict:
    passed = 0
    total = 0
    for r in results:
        criteria = r.get("judge_criteria", {})
        if criteria and criterion in criteria:
            total += 1
            if criteria[criterion].get("passed"):
                passed += 1
    return {
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 3) if total else 0,
    }


def _agg(results: list[dict]) -> dict:
    scores = [r["judge_score"] for r in results if r.get("judge_score") is not None]
    criteria_keys = ["intent", "commands", "analysis", "safety"]

    return {
        "total": len(results),
        "mean_score": round(sum(scores) / len(scores), 3) if scores else 0,
        "by_criterion": {
            crit: _criterion_stats(results, crit) for crit in criteria_keys
        },
    }


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

        by_task[r["task_name"]].append(r)
        by_turn[r["turn"]].append(r)

        mu = r.get("model_usage", {})
        ju = r.get("judge_usage", {})
        total_model_input += mu.get("input_tokens", 0)
        total_model_output += mu.get("output_tokens", 0)
        total_judge_input += ju.get("input_tokens", 0)
        total_judge_output += ju.get("output_tokens", 0)

    scored = [r for r in all_results if not r.get("error")]

    summary = {
        "run_name": config.run_name,
        "model": config.model.model_name,
        "judge": config.judge.model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_rounds": len(all_results),
        "overall": _agg(scored),
        "errors": errors,
        "by_criterion": _agg(scored)["by_criterion"],
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
