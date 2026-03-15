import asyncio
import time
from dataclasses import asdict

from .config import RunConfig
from .data import Sample, load_samples, estimate_tokens
from .judge import judge_compare
from .llm import generate
from .results import init_run_dir, write_sample_result, load_completed_ids, write_summary


async def _eval_sample(
    sample: Sample,
    config: RunConfig,
    semaphore: asyncio.Semaphore,
    progress: dict,
) -> dict:
    async with semaphore:
        start = time.time()
        result = {
            "sample_id": sample.sample_id,
            "task_name": sample.metadata["task_name"],
            "turn": sample.metadata["turn"],
            "total_turns": sample.metadata["total_turns"],
            "reward": sample.metadata.get("reward"),
        }

        try:
            # Generate candidate completion
            candidate, model_usage = await generate(config.model, sample.prompt)
            result["candidate_completion"] = candidate
            result["reference_completion"] = sample.completion
            result["model_usage"] = model_usage

            # Judge comparison
            judge_result = await judge_compare(
                config.judge, sample, sample.completion, candidate
            )
            result["judge_verdict"] = judge_result.verdict
            result["judge_score"] = judge_result.score
            result["judge_reasoning"] = judge_result.reasoning
            result["judge_usage"] = judge_result.usage
            result["error"] = None

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            result["judge_verdict"] = None
            result["judge_score"] = None
            result["judge_reasoning"] = None

        result["duration_sec"] = round(time.time() - start, 2)

        progress["done"] += 1
        status = result.get("judge_verdict", "ERROR")
        print(
            f"  [{progress['done']}/{progress['total']}] "
            f"{sample.metadata['task_name']} turn {sample.metadata['turn']} "
            f"→ {status}"
        )

        return result


def print_summary_table(summary: dict) -> None:
    overall = summary["overall"]
    print(f"\n{'='*60}")
    print(f"Run: {summary['run_name']}")
    print(f"Model: {summary['model']}  |  Judge: {summary['judge']}")
    print(f"{'='*60}")
    print(
        f"Overall: {overall['mean_score']:.1%} mean score "
        f"({overall['equivalent']} eq / {overall['partially_equivalent']} partial / "
        f"{overall['not_equivalent']} not_eq) "
        f"out of {overall['total']} samples"
    )
    if summary.get("errors"):
        print(f"Errors: {summary['errors']}")

    print(f"\n{'Task':<35} {'Score':>6} {'Eq':>4} {'Part':>5} {'Not':>4} {'N':>4}")
    print("-" * 60)
    for task, stats in sorted(summary["by_task"].items()):
        print(
            f"{task:<35} {stats['mean_score']:>5.1%} "
            f"{stats['equivalent']:>4} {stats['partially_equivalent']:>5} "
            f"{stats['not_equivalent']:>4} {stats['total']:>4}"
        )

    usage = summary.get("token_usage", {})
    if any(usage.values()):
        print(f"\nToken usage:")
        print(f"  Model: {usage['model_input_tokens']:,} in / {usage['model_output_tokens']:,} out")
        print(f"  Judge: {usage['judge_input_tokens']:,} in / {usage['judge_output_tokens']:,} out")


async def run(config: RunConfig) -> dict:
    samples = load_samples(config)

    if not samples:
        print("No samples to evaluate.")
        return {}

    run_dir = init_run_dir(config)

    # Check for already-completed samples (resumability)
    completed = load_completed_ids(run_dir)
    pending = [s for s in samples if s.sample_id not in completed]

    if completed:
        print(f"Resuming: {len(completed)} already done, {len(pending)} remaining")

    if not pending:
        print("All samples already completed. Loading existing results.")
        # Reload all results for summary
        all_results = []
        for task_dir in (run_dir / "samples").iterdir():
            if not task_dir.is_dir():
                continue
            for f in task_dir.glob("*.json"):
                import json
                with open(f) as fh:
                    all_results.append(json.load(fh))
        summary = write_summary(run_dir, all_results, config)
        print_summary_table(summary)
        return summary

    print(f"Evaluating {len(pending)} samples (concurrency={config.max_concurrent})")
    semaphore = asyncio.Semaphore(config.max_concurrent)
    progress = {"done": 0, "total": len(pending)}

    tasks = [_eval_sample(s, config, semaphore, progress) for s in pending]
    new_results = await asyncio.gather(*tasks)

    # Write per-sample results
    for result in new_results:
        write_sample_result(run_dir, result)

    # Collect all results (existing + new) for summary
    all_results = list(new_results)
    if completed:
        import json
        for task_dir in (run_dir / "samples").iterdir():
            if not task_dir.is_dir():
                continue
            for f in task_dir.glob("*.json"):
                with open(f) as fh:
                    data = json.load(fh)
                    if data["sample_id"] in completed:
                        all_results.append(data)

    summary = write_summary(run_dir, all_results, config)
    print_summary_table(summary)
    return summary
