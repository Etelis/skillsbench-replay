#!/usr/bin/env python3
"""
Evaluate skillsbench-replay benchmark results using unitxt LLM-as-judge.

Takes a results JSON file produced by run_benchmark.py and scores each
candidate response against its ground-truth reference using four criteria
designed for agent trajectory evaluation:

  - intent:    Does the candidate pursue the same logical next step?
  - commands:  Would the commands produce equivalent results?
  - analysis:  Does the candidate correctly read the task state?
  - safety:    Does the candidate avoid mistakes that derail progress?

All criteria are reference-based — the judge sees both the ground-truth
and the candidate, then issues a binary PASS/FAIL per criterion.

Usage:
    uv run python run_judge.py \
        --results results/bench-baseline-20260328T120000Z.json \
        --judge-endpoint https://api.example.com/v1 \
        --judge-model openai/gpt-oss-120b \
        --api-key YOUR_KEY

    # Compare multiple runs
    uv run python run_judge.py \
        --results results/bench-baseline-*.json results/bench-spans-*.json \
        --judge-endpoint https://api.example.com/v1 \
        --judge-model openai/gpt-oss-120b
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CRITERIA = [
    "intent",
    "commands",
    "analysis",
    "safety",
]

# Custom criteria definitions for agent trajectory evaluation
CRITERIA_DEFINITIONS = {
    "intent": {
        "name": "intent",
        "description": (
            "Evaluate whether the candidate response pursues the same logical "
            "next step as the reference response. Both should attempt the same "
            "goal or action. PASS if they share the same intent; FAIL if the "
            "candidate pursues a different goal or skips a step."
        ),
        "options": [
            {"name": "PASS", "description": "Both responses attempt the same logical next step"},
            {"name": "FAIL", "description": "Candidate pursues a different goal or skips a step"},
        ],
        "option_map": {"PASS": 1.0, "FAIL": 0.0},
    },
    "commands": {
        "name": "commands",
        "description": (
            "Evaluate whether the commands in the candidate response would "
            "produce equivalent results to the reference commands. Minor "
            "differences in flags or ordering are acceptable if the outcome "
            "is the same. PASS if equivalent; FAIL if materially different."
        ),
        "options": [
            {"name": "PASS", "description": "Commands would produce equivalent results"},
            {"name": "FAIL", "description": "Commands would produce materially different results"},
        ],
        "option_map": {"PASS": 1.0, "FAIL": 0.0},
    },
    "analysis": {
        "name": "analysis",
        "description": (
            "Evaluate whether the candidate correctly reads and interprets "
            "the current task state shown in the terminal output. PASS if "
            "the candidate's analysis is accurate; FAIL if it misinterprets "
            "output, errors, or prior results."
        ),
        "options": [
            {"name": "PASS", "description": "Candidate correctly reads the current task state"},
            {"name": "FAIL", "description": "Candidate misinterprets output, errors, or prior results"},
        ],
        "option_map": {"PASS": 1.0, "FAIL": 0.0},
    },
    "safety": {
        "name": "safety",
        "description": (
            "Evaluate whether the candidate avoids mistakes that would derail "
            "the task or block future progress. Compare against the reference "
            "to check for dangerous deviations. PASS if safe; FAIL if the "
            "candidate introduces errors that break the task."
        ),
        "options": [
            {"name": "PASS", "description": "No mistakes that would derail task progress"},
            {"name": "FAIL", "description": "Introduces errors that break the task or block future progress"},
        ],
        "option_map": {"PASS": 1.0, "FAIL": 0.0},
    },
}


def judge_results(results_file: Path, args) -> dict:
    """Run unitxt LLM-as-judge on a single results file."""
    from unitxt.api import create_dataset, evaluate
    from unitxt.inference import OpenAiInferenceEngine
    from unitxt.llm_as_judge import LLMJudgeDirect
    from unitxt.llm_as_judge_constants import CriteriaWithOptions, CriteriaOption

    with open(results_file) as f:
        results = json.load(f)

    samples = results["samples"]
    valid = [s for s in samples if s.get("candidate") and "error" not in s]

    if not valid:
        print(f"  No valid samples to judge in {results_file.name}")
        return {}

    # Build criteria objects
    criteria_objects = {}
    for name, defn in CRITERIA_DEFINITIONS.items():
        criteria_objects[name] = CriteriaWithOptions(
            name=defn["name"],
            description=defn["description"],
            options=[
                CriteriaOption(name=o["name"], description=o["description"])
                for o in defn["options"]
            ],
            option_map=defn["option_map"],
        )

    # Build unitxt data — reference goes into 'context'
    data = [
        {
            "question": s["sample_id"],
            "context": s["reference"],
            "context_type": "text",
        }
        for s in valid
    ]
    predictions = [s["candidate"] for s in valid]

    # Inference engine
    api_key = args.api_key or "EMPTY"
    default_headers = {}
    if api_key and api_key != "EMPTY":
        default_headers["RITS_API_KEY"] = api_key
    engine = OpenAiInferenceEngine(
        model_name=args.judge_model,
        base_url=args.judge_endpoint,
        credentials={"api_key": api_key, "api_url": args.judge_endpoint},
        default_headers=default_headers,
        max_tokens=1024,
        temperature=0.0,
    )

    # Run each criterion separately to isolate failures
    criterion_scores = {}
    all_instance_scores = [{} for _ in valid]

    for crit_name in CRITERIA:
        print(f"  Evaluating criterion: {crit_name}")
        try:
            metric = LLMJudgeDirect(
                inference_engine=engine,
                criteria=criteria_objects[crit_name],
                context_fields=["context"],
                criteria_field="criteria",
            )

            dataset = create_dataset(
                task="tasks.qa.with_context",
                test_set=data,
                metrics=[metric],
                split="test",
            )

            eval_results = evaluate(predictions=predictions, data=dataset)

            # Extract global score
            for k, v in eval_results.global_scores.items():
                if crit_name in k and isinstance(v, (int, float)):
                    criterion_scores[crit_name] = v
                    break

            # Extract per-instance scores
            for i, inst in enumerate(eval_results.instance_scores):
                for k, v in inst.items():
                    if crit_name in str(k) and isinstance(v, (int, float)):
                        all_instance_scores[i][crit_name] = v
                        break
        except Exception as e:
            print(f"  WARNING: criterion '{crit_name}' failed: {e}")
            criterion_scores[crit_name] = 0.0

    instance_scores = [
        {"sample_id": valid[i]["sample_id"], **all_instance_scores[i]}
        for i in range(len(valid))
    ]

    # Per-task aggregation
    task_scores = {}
    for s, inst in zip(valid, instance_scores):
        task = s["task"]
        if task not in task_scores:
            task_scores[task] = {c: [] for c in CRITERIA}
        for c in CRITERIA:
            if c in inst:
                task_scores[task][c].append(inst[c])

    by_task = {}
    for task, scores in task_scores.items():
        by_task[task] = {
            c: sum(v) / len(v) if v else 0.0
            for c, v in scores.items()
        }
        by_task[task]["overall"] = (
            sum(by_task[task][c] for c in CRITERIA) / len(CRITERIA)
        )

    overall = (
        sum(criterion_scores.values()) / len(criterion_scores)
        if criterion_scores else 0.0
    )

    return {
        "source_file": results_file.name,
        "label": results.get("label"),
        "model": results.get("model"),
        "judge_model": args.judge_model,
        "num_judged": len(valid),
        "overall_score": overall,
        "by_criterion": criterion_scores,
        "by_task": by_task,
        "instance_scores": instance_scores,
    }


def print_summary(judge_outputs: list[dict]):
    """Print comparison table."""
    print(f"\n{'=' * 70}")
    print("LLM-as-Judge Results (skillsbench-replay)")
    print(f"{'=' * 70}")

    if len(judge_outputs) == 1:
        r = judge_outputs[0]
        print(f"Model:    {r['model']}")
        print(f"Label:    {r.get('label', 'default')}")
        print(f"Judge:    {r['judge_model']}")
        print(f"Judged:   {r['num_judged']} samples")
        print(f"Overall:  {r['overall_score']:.1%}")
        print()
        for k, v in r.get("by_criterion", {}).items():
            print(f"  {k:<12s}: {v:.1%}")
        print()
        if r.get("by_task"):
            print(f"{'Task':<35s} {'Overall':>8s} {'Intent':>8s} {'Cmds':>8s} {'Anlys':>8s} {'Safety':>8s}")
            print("-" * 75)
            for task, scores in sorted(r["by_task"].items()):
                print(
                    f"{task:<35s} {scores.get('overall', 0):>7.1%} "
                    f"{scores.get('intent', 0):>7.1%} "
                    f"{scores.get('commands', 0):>7.1%} "
                    f"{scores.get('analysis', 0):>7.1%} "
                    f"{scores.get('safety', 0):>7.1%}"
                )
    else:
        header = f"{'Label':<20s}"
        for c in CRITERIA:
            header += f" {c:>10s}"
        header += f" {'Overall':>10s}"
        print(header)
        print("-" * len(header))

        for r in judge_outputs:
            label = r.get("label", "default") or "default"
            row = f"{label:<20s}"
            for c in CRITERIA:
                score = r.get("by_criterion", {}).get(c, 0)
                row += f" {score:>9.1%}"
            row += f" {r['overall_score']:>9.1%}"
            print(row)

    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate skillsbench-replay results with LLM-as-judge"
    )
    parser.add_argument(
        "--results", nargs="+", required=True,
        help="Path(s) to benchmark result JSON files",
    )
    parser.add_argument("--judge-endpoint", required=True, help="Judge model endpoint")
    parser.add_argument("--judge-model", required=True, help="Judge model name")
    parser.add_argument("--api-key", default=None, help="API key for judge endpoint")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    results_files = []
    for pattern in args.results:
        p = Path(pattern)
        if p.exists():
            results_files.append(p)
        else:
            matched = list(Path(".").glob(pattern))
            results_files.extend(sorted(matched))

    if not results_files:
        print("No results files found.", file=sys.stderr)
        sys.exit(1)

    judge_outputs = []
    for rf in results_files:
        print(f"\nJudging: {rf.name}")
        output = judge_results(rf, args)
        if output:
            judge_outputs.append(output)

    if not judge_outputs:
        print("No results to report.", file=sys.stderr)
        sys.exit(1)

    print_summary(judge_outputs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = output_dir / f"judge-{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(judge_outputs, f, indent=2)
    print(f"\nJudge results: {out_file}")


if __name__ == "__main__":
    main()
