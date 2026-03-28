#!/usr/bin/env python3
"""
Replay skillsbench-replay prompts against a target LLM endpoint.

Sends each sample's prompt (full conversation history) to the model and
records the response alongside latency and token-usage metrics.

When the endpoint is a spans middleware, the response may include
spans_metadata which is captured per-sample in the output.

Output is written to a timestamped JSON file under --output-dir.

Usage:
    uv run python run_benchmark.py \
        --endpoint http://localhost:9000/v1 \
        --model meta-llama/Llama-3.1-8B-Instruct

    # Cap to 10 samples for a smoke test
    uv run python run_benchmark.py \
        --endpoint http://localhost:9000/v1 \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --max-samples 10

    # Only run a specific task
    uv run python run_benchmark.py \
        --endpoint http://localhost:9000/v1 \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --task dialogue-parser
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BENCHMARK_DIR = Path(__file__).resolve().parent / "benchmark"


async def generate_single(client, endpoint, model, messages, temperature, max_tokens, seed, sem):
    """Send a chat completion request and capture the full response."""
    async with sem:
        try:
            start = time.time()
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": 1.0,
                "max_tokens": max_tokens,
            }
            if seed is not None:
                payload["seed"] = seed
            resp = await client.post(
                f"{endpoint}/chat/completions",
                json=payload,
                timeout=300.0,
            )
            latency = time.time() - start
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            usage = data.get("usage", {})
            spans_metadata = data.get("spans_metadata")

            result = {
                "content": choice.get("message", {}).get("content", ""),
                "finish_reason": choice.get("finish_reason"),
                "latency_seconds": latency,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
            if spans_metadata:
                result["spans_metadata"] = spans_metadata

            return result
        except Exception as e:
            return {"content": "", "error": str(e)}


def load_samples(task_filter=None, max_samples=None):
    """Load all samples from benchmark directory, optionally filtered by task."""
    samples = []
    task_dirs = sorted(
        d for d in BENCHMARK_DIR.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    )

    for task_dir in task_dirs:
        task_name = task_dir.name
        if task_filter and task_name != task_filter:
            continue

        sample_dirs = sorted(
            d for d in task_dir.iterdir()
            if d.is_dir() and d.name.startswith("sample_")
        )

        for sample_dir in sample_dirs:
            prompt_file = sample_dir / "prompt.json"
            response_file = sample_dir / "response.json"
            if not prompt_file.exists() or not response_file.exists():
                continue

            prompt = json.loads(prompt_file.read_text())
            reference = json.loads(response_file.read_text())

            samples.append({
                "task": task_name,
                "sample_id": f"{task_name}/{sample_dir.name}",
                "messages": prompt["messages"],
                "reference": reference["content"],
            })

    if max_samples:
        samples = samples[:max_samples]

    return samples


async def run(args):
    samples = load_samples(task_filter=args.task, max_samples=args.max_samples)
    if not samples:
        raise SystemExit("No samples found.")

    api_key = args.api_key or os.environ.get("API_KEY", "EMPTY")
    headers = {"Authorization": f"Bearer {api_key}"}
    if api_key and api_key != "EMPTY":
        headers["RITS_API_KEY"] = api_key

    async with httpx.AsyncClient(headers=headers, verify=False) as client:
        sem = asyncio.Semaphore(args.max_concurrent)

        tasks_set = sorted(set(s["task"] for s in samples))
        print(f"Tasks:    {len(tasks_set)} ({', '.join(tasks_set)})")
        print(f"Samples:  {len(samples)}")
        print(f"Model:    {args.model}")
        print(f"Endpoint: {args.endpoint}")
        print()

        tasks = []
        for s in samples:
            tasks.append(
                generate_single(
                    client, args.endpoint, args.model,
                    s["messages"],
                    args.temperature, args.max_tokens, args.seed,
                    sem,
                )
            )

        print("Generating responses...")
        results = await asyncio.gather(*tasks)

    n_ok = sum(1 for r in results if "error" not in r)
    n_err = len(results) - n_ok
    total_latency = sum(r.get("latency_seconds", 0) for r in results if "error" not in r)
    total_prompt = sum(r.get("prompt_tokens", 0) for r in results if "error" not in r)
    total_completion = sum(r.get("completion_tokens", 0) for r in results if "error" not in r)

    per_sample = []
    for s, res in zip(samples, results):
        entry = {
            "task": s["task"],
            "sample_id": s["sample_id"],
            "reference": s["reference"],
            "candidate": res.get("content", ""),
            "finish_reason": res.get("finish_reason"),
            "latency_seconds": res.get("latency_seconds", 0),
            "prompt_tokens": res.get("prompt_tokens", 0),
            "completion_tokens": res.get("completion_tokens", 0),
        }
        if "spans_metadata" in res:
            entry["spans_metadata"] = res["spans_metadata"]
        if "error" in res:
            entry["error"] = res["error"]
        per_sample.append(entry)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = args.label or "default"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "benchmark": "skillsbench-replay",
        "timestamp": timestamp,
        "label": label,
        "model": args.model,
        "endpoint": args.endpoint,
        "metrics": {
            "total_samples": len(samples),
            "generated": n_ok,
            "errors": n_err,
            "avg_latency_seconds": total_latency / n_ok if n_ok else 0,
            "total_latency_seconds": total_latency,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
        },
        "samples": per_sample,
    }

    out_file = output_dir / f"bench-{label}-{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Model:      {args.model}")
    print(f"Samples:    {n_ok}/{len(samples)} ({n_err} errors)")
    if label != "default":
        print(f"Label:      {label}")
    print(f"Avg latency: {total_latency / n_ok:.2f}s" if n_ok else "Avg latency: N/A")
    print(f"Tokens:     {total_prompt:,} prompt / {total_completion:,} completion")
    spans_count = sum(1 for s in per_sample if "spans_metadata" in s)
    if spans_count:
        print(f"Spans:      {spans_count}/{n_ok} samples with middleware metadata")
    print(f"{'=' * 60}")
    print(f"\nResults: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Replay skillsbench prompts against an LLM endpoint"
    )
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--task", default=None, help="Run only a specific task")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit total samples")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max parallel requests")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (None to disable)")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--label", default=None, help="Run label (e.g. baseline, spans, naive)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
