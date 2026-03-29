#!/usr/bin/env python3
"""
Evaluate skillsbench-replay benchmark results using unitxt LLM-as-judge.

Two modes:

1. **Task evaluation** (default): Scores each response against the Haiku
   ground-truth using 4 agent-trajectory criteria (intent, commands,
   analysis, safety).

2. **Spans comparison** (--baseline): Scores spans/naive outputs against
   the baseline output using correctness + consistency criteria. This
   measures whether the middleware degrades quality.

Usage:
    # Task evaluation
    uv run python run_judge.py \
        --results results/bench-baseline-*.json \
        --judge-endpoint https://api.example.com/v1 \
        --judge-model openai/gpt-oss-120b

    # Spans comparison (baseline as ground truth)
    uv run python run_judge.py \
        --results results/bench-spans-*.json results/bench-naive-*.json \
        --baseline results/bench-baseline-20260328T201348Z.json \
        --judge-endpoint https://api.example.com/v1 \
        --judge-model openai/gpt-oss-120b
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Task evaluation criteria (custom, binary PASS/FAIL) ──────────────

TASK_CRITERIA = ["intent", "commands", "analysis", "safety"]

TASK_CRITERIA_DEFINITIONS = {
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

# ── Spans comparison criteria (unitxt catalog, reference-based) ───────

COMPARISON_CRITERIA = [
    "correctness_based_on_ground_truth",
    "consistency",
]


def _build_baseline_index(baseline_file: Path) -> dict:
    """Build sample_id -> candidate mapping from a baseline results file."""
    with open(baseline_file) as f:
        baseline = json.load(f)
    index = {}
    for s in baseline["samples"]:
        if s.get("candidate") and "error" not in s:
            index[s["sample_id"]] = s["candidate"]
    return index


def judge_task_eval(results_file: Path, args) -> dict:
    """Task evaluation: judge against Haiku ground-truth with 4 criteria."""
    from unitxt.api import create_dataset, evaluate
    from unitxt.inference import OpenAiInferenceEngine
    from unitxt.llm_as_judge import LLMJudgeDirect
    from unitxt.llm_as_judge_constants import CriteriaWithOptions, CriteriaOption

    with open(results_file) as f:
        results = json.load(f)

    samples = results["samples"]
    valid = [s for s in samples if s.get("candidate") and "error" not in s]

    if not valid:
        print(f"  No valid samples in {results_file.name}")
        return {}

    criteria_objects = {}
    for name, defn in TASK_CRITERIA_DEFINITIONS.items():
        criteria_objects[name] = CriteriaWithOptions(
            name=defn["name"],
            description=defn["description"],
            options=[
                CriteriaOption(name=o["name"], description=o["description"])
                for o in defn["options"]
            ],
            option_map=defn["option_map"],
        )

    data = [
        {"question": s["sample_id"], "context": s["reference"], "context_type": "text"}
        for s in valid
    ]
    predictions = [s["candidate"] for s in valid]

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

    criteria_names = TASK_CRITERIA
    criterion_scores = {}
    all_instance_scores = [{} for _ in valid]

    for crit_name in criteria_names:
        print(f"  Evaluating criterion: {crit_name}")
        try:
            metric = LLMJudgeDirect(
                inference_engine=engine,
                criteria=criteria_objects[crit_name],
                context_fields=["context"],
                criteria_field="criteria",
            )
            dataset = create_dataset(
                task="tasks.qa.with_context", test_set=data,
                metrics=[metric], split="test",
            )
            eval_results = evaluate(predictions=predictions, data=dataset)

            for k, v in eval_results.global_scores.items():
                if crit_name in k and isinstance(v, (int, float)):
                    criterion_scores[crit_name] = v
                    break
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

    overall = (
        sum(criterion_scores.values()) / len(criterion_scores)
        if criterion_scores else 0.0
    )

    return {
        "mode": "task_eval",
        "source_file": results_file.name,
        "label": results.get("label"),
        "model": results.get("model"),
        "judge_model": args.judge_model,
        "num_judged": len(valid),
        "overall_score": overall,
        "by_criterion": criterion_scores,
        "instance_scores": instance_scores,
    }


def judge_comparison(results_file: Path, baseline_index: dict, args) -> dict:
    """Spans comparison: judge against baseline output with correctness + consistency."""
    from unitxt.api import create_dataset, evaluate
    from unitxt.inference import OpenAiInferenceEngine
    from unitxt.llm_as_judge import LLMJudgeDirect

    with open(results_file) as f:
        results = json.load(f)

    samples = results["samples"]
    valid = []
    for s in samples:
        if not s.get("candidate") or "error" in s:
            continue
        sid = s["sample_id"]
        if sid not in baseline_index:
            continue
        valid.append(s)

    if not valid:
        print(f"  No valid samples in {results_file.name}")
        return {}

    # Reference = baseline output, prediction = spans/naive output
    data = [
        {
            "question": s["sample_id"],
            "context": baseline_index[s["sample_id"]],
            "context_type": "text",
        }
        for s in valid
    ]
    predictions = [s["candidate"] for s in valid]

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

    criterion_scores = {}
    all_instance_scores = [{} for _ in valid]

    for crit_name in COMPARISON_CRITERIA:
        print(f"  Evaluating criterion: {crit_name}")
        try:
            criterion_ref = f"metrics.llm_as_judge.direct.criteria.{crit_name}"
            metric = LLMJudgeDirect(
                inference_engine=engine,
                criteria=criterion_ref,
                context_fields=["context"],
                criteria_field="criteria",
            )
            dataset = create_dataset(
                task="tasks.qa.with_context", test_set=data,
                metrics=[metric], split="test",
            )
            eval_results = evaluate(predictions=predictions, data=dataset)

            for k, v in eval_results.global_scores.items():
                if crit_name in k and isinstance(v, (int, float)):
                    criterion_scores[crit_name] = v
                    break
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

    overall = (
        sum(criterion_scores.values()) / len(criterion_scores)
        if criterion_scores else 0.0
    )

    return {
        "mode": "comparison",
        "source_file": results_file.name,
        "label": results.get("label"),
        "model": results.get("model"),
        "judge_model": args.judge_model,
        "baseline_file": args.baseline,
        "num_judged": len(valid),
        "overall_score": overall,
        "by_criterion": criterion_scores,
        "instance_scores": instance_scores,
    }


def print_summary(judge_outputs: list[dict]):
    """Print comparison table."""
    if not judge_outputs:
        return

    mode = judge_outputs[0].get("mode", "task_eval")
    criteria = COMPARISON_CRITERIA if mode == "comparison" else TASK_CRITERIA

    print(f"\n{'=' * 70}")
    if mode == "comparison":
        print("LLM-as-Judge: Spans Comparison (vs baseline)")
    else:
        print("LLM-as-Judge: Task Evaluation (vs Haiku reference)")
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
            print(f"  {k:<45s}: {v:.3f}")
    else:
        header = f"{'Label':<20s} {'Model':<30s}"
        for c in criteria:
            short = c[:15]
            header += f" {short:>15s}"
        header += f" {'Overall':>10s}"
        print(header)
        print("-" * len(header))

        for r in judge_outputs:
            label = r.get("label", "default") or "default"
            model = (r.get("model") or "?").split("/")[-1][:28]
            row = f"{label:<20s} {model:<30s}"
            for c in criteria:
                score = r.get("by_criterion", {}).get(c, 0)
                row += f" {score:>14.1%}"
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
    parser.add_argument(
        "--baseline", default=None,
        help="Baseline results file. When provided, uses correctness + consistency "
             "to compare spans/naive against baseline (instead of task criteria).",
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

    baseline_index = None
    if args.baseline:
        baseline_index = _build_baseline_index(Path(args.baseline))
        print(f"Baseline: {args.baseline} ({len(baseline_index)} samples)")

    judge_outputs = []
    for rf in results_files:
        print(f"\nJudging: {rf.name}")
        if baseline_index is not None:
            output = judge_comparison(rf, baseline_index, args)
        else:
            output = judge_task_eval(rf, args)
        if output:
            judge_outputs.append(output)

    if not judge_outputs:
        print("No results to report.", file=sys.stderr)
        sys.exit(1)

    print_summary(judge_outputs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "comparison" if baseline_index else "task-eval"
    out_file = output_dir / f"judge-{suffix}-{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(judge_outputs, f, indent=2)
    print(f"\nJudge results: {out_file}")


if __name__ == "__main__":
    main()
