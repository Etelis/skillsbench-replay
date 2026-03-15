#!/usr/bin/env python3
"""
Trajectory-Bench: Evaluate LLM response equivalence on solved agent trajectories.

Usage:
    python run.py --config configs/eval-haiku.yaml
    python run.py --config configs/eval-haiku.yaml --dry-run
    python run.py --config configs/eval-haiku.yaml --max-samples 5
"""

import argparse
import asyncio
import sys

from trajectory_bench.config import load_config
from trajectory_bench.data import load_samples, estimate_tokens
from trajectory_bench.runner import run


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
    total = sum(est.values()) - est["n_samples"]
    print(f"  Total:        {total:>10,}")


def main():
    parser = argparse.ArgumentParser(description="Trajectory-Bench evaluation runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Load data and print stats without calling LLMs")
    parser.add_argument("--max-samples", type=int, help="Override max_samples in config")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.max_samples is not None:
        config.max_samples = args.max_samples

    if args.dry_run:
        dry_run(config)
        return

    asyncio.run(run(config))


if __name__ == "__main__":
    main()
