import asyncio
import json
import time

from .config import RunConfig
from .data import Round, load_rounds, estimate_tokens
from .judge import judge_compare, CRITERIA
from .llm import generate
from .results import init_run_dir, write_round_result, load_completed_ids, write_summary


async def _eval_round(
    round_data: Round,
    config: RunConfig,
    semaphore: asyncio.Semaphore,
    progress: dict,
) -> dict:
    async with semaphore:
        start = time.time()
        result = {
            "round_id": round_data.round_id,
            "task_name": round_data.metadata["task_name"],
            "turn": round_data.metadata["turn"],
            "total_turns": round_data.metadata["total_turns"],
            "reward": round_data.metadata.get("reward"),
        }

        try:
            # Generate candidate completion
            candidate, model_usage = await generate(config.model, round_data.prompt)
            result["candidate_completion"] = candidate
            result["reference_completion"] = round_data.completion
            result["model_usage"] = model_usage

            # Judge comparison — 4 independent criterion calls
            judge_result = await judge_compare(
                config.judge, round_data, round_data.completion, candidate
            )

            result["judge_score"] = judge_result.score
            result["judge_criteria"] = {
                k: {
                    "verdict": r.verdict,
                    "passed": r.passed,
                    "reasoning": r.reasoning,
                }
                for k, r in judge_result.criteria.items()
            }
            result["judge_usage"] = judge_result.usage
            result["error"] = None

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            result["judge_score"] = None
            result["judge_criteria"] = None

        result["duration_sec"] = round(time.time() - start, 2)

        progress["done"] += 1
        if result.get("judge_criteria"):
            passed = [k for k, r in result["judge_criteria"].items() if r["passed"]]
            status = f"{len(passed)}/4 ({', '.join(passed) or 'none'})"
        else:
            status = "ERROR"
        print(
            f"  [{progress['done']}/{progress['total']}] "
            f"{round_data.metadata['task_name']} turn {round_data.metadata['turn']} "
            f"→ {status}"
        )

        return result


def print_summary_table(summary: dict) -> None:
    overall = summary["overall"]
    print(f"\n{'='*70}")
    print(f"Run: {summary['run_name']}")
    print(f"Model: {summary['model']}  |  Judge: {summary['judge']}")
    print(f"{'='*70}")
    print(f"Overall: {overall['mean_score']:.1%} mean score across {overall['total']} rounds")

    # Per-criterion pass rates
    by_criterion = summary.get("by_criterion", {})
    if by_criterion:
        print(f"\nPer-criterion pass rates:")
        for crit, stats in by_criterion.items():
            print(f"  {crit:<12} {stats['pass_rate']:.1%} ({stats['passed']}/{stats['total']})")

    if summary.get("errors"):
        print(f"Errors: {summary['errors']}")

    print(f"\n{'Task':<35} {'Score':>6} {'Intent':>7} {'Cmds':>6} {'Anlys':>7} {'Safe':>6} {'N':>4}")
    print("-" * 75)
    for task, stats in sorted(summary["by_task"].items()):
        tc = stats.get("by_criterion", {})
        print(
            f"{task:<35} {stats['mean_score']:>5.1%} "
            f"{tc.get('intent', {}).get('pass_rate', 0):>6.0%} "
            f"{tc.get('commands', {}).get('pass_rate', 0):>6.0%} "
            f"{tc.get('analysis', {}).get('pass_rate', 0):>6.0%} "
            f"{tc.get('safety', {}).get('pass_rate', 0):>6.0%} "
            f"{stats['total']:>4}"
        )

    usage = summary.get("token_usage", {})
    if any(usage.values()):
        print(f"\nToken usage:")
        print(f"  Model: {usage['model_input_tokens']:,} in / {usage['model_output_tokens']:,} out")
        print(f"  Judge: {usage['judge_input_tokens']:,} in / {usage['judge_output_tokens']:,} out")


async def run(config: RunConfig) -> dict:
    rounds = load_rounds(config)

    if not rounds:
        print("No rounds to evaluate.")
        return {}

    run_dir = init_run_dir(config)

    # Check for already-completed rounds (resumability)
    completed = load_completed_ids(run_dir)
    pending = [r for r in rounds if r.round_id not in completed]

    if completed:
        print(f"Resuming: {len(completed)} already done, {len(pending)} remaining")

    if not pending:
        print("All rounds already completed. Loading existing results.")
        all_results = []
        for task_dir in (run_dir / "rounds").iterdir():
            if not task_dir.is_dir():
                continue
            for f in task_dir.glob("*.json"):
                with open(f) as fh:
                    all_results.append(json.load(fh))
        summary = write_summary(run_dir, all_results, config)
        print_summary_table(summary)
        return summary

    print(f"Evaluating {len(pending)} rounds (concurrency={config.max_concurrent})")
    semaphore = asyncio.Semaphore(config.max_concurrent)
    progress = {"done": 0, "total": len(pending)}

    tasks = [_eval_round(r, config, semaphore, progress) for r in pending]
    new_results = await asyncio.gather(*tasks)

    # Write per-round results
    for result in new_results:
        write_round_result(run_dir, result)

    # Collect all results (existing + new) for summary
    all_results = list(new_results)
    if completed:
        for task_dir in (run_dir / "rounds").iterdir():
            if not task_dir.is_dir():
                continue
            for f in task_dir.glob("*.json"):
                with open(f) as fh:
                    data = json.load(fh)
                    if data["round_id"] in completed:
                        all_results.append(data)

    summary = write_summary(run_dir, all_results, config)
    print_summary_table(summary)
    return summary
