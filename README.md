# skillsbench-replay

Evaluate small and lightweight models on [SkillsBench](https://github.com/harbor-ai/skillsbench) tasks — without running any tools or environments.

## Motivation

Running SkillsBench end-to-end requires Docker containers, tool execution, and environment setup for each task. This makes it expensive and slow to evaluate smaller or self-hosted models.

**skillsbench-replay** takes a different approach: we first ran SkillsBench with Haiku 4.5 (itself a relatively weak model) and kept only the tasks it solved successfully. From those solved runs, we extracted the full agent trajectories — every prompt the model received and every response it produced.

Now, to evaluate a new model, we simply **replay the prompts** and use an **LLM-as-judge** to check if the new model's responses are functionally equivalent to the known-good ones. No tools, no Docker, no environment setup — just inference.

This lets you benchmark models that are too small or too slow for full agentic evaluation, including locally-served models via **vLLM** or any OpenAI-compatible endpoint.

## Quick start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install anthropic openai pyyaml tenacity

# Dry run (no API calls, shows sample count + estimated tokens)
python run.py --config configs/eval-haiku.yaml --dry-run

# Run on a few samples
ANTHROPIC_API_KEY="..." python run.py --config configs/eval-haiku.yaml --max-samples 5

# Full run
ANTHROPIC_API_KEY="..." python run.py --config configs/eval-haiku.yaml
```

### With vLLM

Serve your model:
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

Evaluate it:
```bash
ANTHROPIC_API_KEY="..." python run.py --config configs/eval-vllm.yaml
```

The eval model hits your local vLLM server while the judge uses Anthropic's API (or you can point the judge at a vLLM instance too).

## How it works

```
For each of the 181 samples:
  1. Send the prompt (conversation history) to the eval model
  2. Get the candidate completion
  3. Send reference + candidate to the judge model
  4. Judge scores: equivalent (1.0) | partially_equivalent (0.5) | not_equivalent (0.0)
```

The judge evaluates on four criteria:
- **Intent**: Same logical next step?
- **Commands**: Functionally equivalent outcomes?
- **Analysis**: Comparable understanding of task state?
- **Errors**: Any mistakes that would derail the task?

## Configuration

```yaml
run_name: eval-vllm-llama-8b

model:
  provider: openai              # "anthropic" or "openai" (vLLM, TGI, etc.)
  model_name: meta-llama/Llama-3.1-8B-Instruct
  base_url: http://localhost:8000/v1   # vLLM endpoint
  temperature: 0.7
  max_tokens: 8192

judge:
  provider: anthropic
  model_name: claude-sonnet-4-5-20250929
  temperature: 0.0
  max_tokens: 2048

data_path: data/samples/all-solved.jsonl
filter_tasks: null        # e.g. ["dialogue-parser", "flood-risk-analysis"]
filter_turns: null        # e.g. [0] for first turn only
max_samples: null         # cap total samples

max_concurrent: 10
output_dir: results
max_retries: 2
```

The `openai` provider works with any OpenAI-compatible API: vLLM, TGI, Ollama, Together, etc. Set `base_url` to point at your endpoint.

## Results

Results are written to `results/{run_name}/`:

```
results/eval-vllm-llama-8b/
├── config.yaml                          # Config snapshot
├── summary.json                         # Aggregate scores (overall, by-task, by-turn)
└── samples/
    └── {task_name}/
        └── {sample_id}.json             # Per-sample: candidate, verdict, reasoning
```

Runs are **resumable** — restarting with the same config skips already-completed samples.

## Data

| Directory | Contents |
|-----------|----------|
| `data/samples/all-solved.jsonl` | 181 prompt/completion samples (8.5MB) |
| `data/raw-trajectories/` | Original agent trajectories from 13 solved tasks |
| `scripts/trajectory_to_samples.py` | Script to convert trajectories into JSONL samples |

### Tasks included (13 tasks, 181 samples)

All tasks were successfully solved by Haiku 4.5 on SkillsBench.

| Task | Samples | Reward |
|------|---------|--------|
| protein-expression-analysis | 33 | 1.000 |
| hvac-control | 21 | 1.000 |
| weighted-gdp-calc | 20 | 1.000 |
| mario-coin-counting | 18 | 1.000 |
| lab-unit-harmonization | 16 | 0.354 |
| virtualhome-agent-planning | 15 | 1.000 |
| econ-detrending-correlation | 14 | 1.000 |
| dialogue-parser | 12 | 0.833 |
| flood-risk-analysis | 8 | 1.000 |
| video-filler-word-remover | 7 | 1.000 |
| offer-letter-generator | 6 | 1.000 |
| threejs-structure-parser | 6 | 1.000 |
| gravitational-wave-detection | 5 | 1.000 |

## Estimated cost

A full run (181 samples) uses roughly:
- ~2M model input tokens + ~100K model output tokens
- ~290K judge input tokens + ~18K judge output tokens

For locally-served models (vLLM), only the judge tokens incur API cost.
