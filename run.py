#!/usr/bin/env python3
"""
skillsbench-replay: Evaluate LLM response equivalence on solved agent trajectories.

Usage (kvcache-bench style):
    python run.py --endpoint http://localhost:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct

Usage (YAML config):
    python run.py --config configs/eval-vllm.yaml

Usage (dry run):
    python run.py --endpoint http://localhost:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct --dry-run
"""

import argparse
import asyncio
import sys
from pathlib import Path

from trajectory_bench.config import load_config, RunConfig, ModelConfig, JudgeConfig
from trajectory_bench.data import load_samples, estimate_tokens
from trajectory_bench.runner import run


def config_from_cli(args) -> RunConfig:
    """Build RunConfig from CLI args (kvcache-bench compatible)."""
    model_name = args.model.rstrip("/")
    short_name = model_name.split("/")[-1].lower()

    model = ModelConfig(
        provider="openai",
        model_name=model_name,
        base_url=args.endpoint,
        temperature=0.7,
        max_tokens=8192,
    )

    # Judge defaults to same endpoint/model unless overridden
    judge = JudgeConfig(
        provider="openai",
        model_name=args.judge_model or model_name,
        base_url=args.judge_endpoint or args.endpoint,
        temperature=0.0,
        max_tokens=2048,
    )

    # Resolve data_path relative to this script
    script_dir = Path(__file__).resolve().parent
    data_path = str(script_dir / "data" / "samples" / "all-solved.jsonl")

    return RunConfig(
        run_name=f"eval-{short_name}",
        model=model,
        judge=judge,
        data_path=data_path,
        max_samples=args.max_samples,
        max_concurrent=args.max_concurrent,
        output_dir=args.output_dir,
    )


def dry_run(config):
    samples = load_samples(config)
    est = estimate_tokens(samples)

    tasks = {}
    for s in samples:
        t = s.metadata["task_name"]
        tasks[t] = tasks.get(t, 0) + 1

    print(f"Dry run — {est['n_samples']} samples loaded")
    print(f"\nTask breakdown:")
    for task, count in sorted(tasks.items()):
        print(f"  {task:<35} {count:>4} samples")

    print(f"\nEstimated tokens:")
    print(f"  Model input:  {est['est_model_input_tokens']:>10,}")
    print(f"  Model output: {est['est_model_output_tokens']:>10,}")
    print(f"  Judge input:  {est['est_judge_input_tokens']:>10,}")
    print(f"  Judge output: {est['est_judge_output_tokens']:>10,}")


def main():
    parser = argparse.ArgumentParser(description="skillsbench-replay evaluation")

    # kvcache-bench compatible args
    parser.add_argument("--endpoint", help="vLLM API endpoint (e.g. http://localhost:8000/v1)")
    parser.add_argument("--model", help="Model name (e.g. meta-llama/Llama-3.1-8B-Instruct)")

    # Optional judge override
    parser.add_argument("--judge-endpoint", help="Judge vLLM endpoint (defaults to --endpoint)")
    parser.add_argument("--judge-model", help="Judge model name (defaults to --model)")

    # YAML config (alternative to CLI args)
    parser.add_argument("--config", help="Path to YAML config file")

    # Common options
    parser.add_argument("--dry-run", action="store_true", help="Print stats without calling LLMs")
    parser.add_argument("--max-samples", type=int, help="Cap number of samples")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("--output-dir", default="results", help="Output directory")

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        if args.max_samples is not None:
            config.max_samples = args.max_samples
    elif args.endpoint and args.model:
        config = config_from_cli(args)
    else:
        parser.error("Either --config or both --endpoint and --model are required")

    if args.dry_run:
        dry_run(config)
        return

    asyncio.run(run(config))


if __name__ == "__main__":
    main()
