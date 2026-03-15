# skillsbench-replay

Evaluate whether an LLM produces functionally equivalent responses when replaying solved [SkillsBench](https://github.com/harbor-ai/skillsbench) agent trajectories.

## What is this?

SkillsBench tasks are solved by an LLM agent over multiple turns (prompt → action → observation → ...). From 13 successfully solved tasks (Haiku 4.5), we extracted **181 prompt/completion samples** — one per agent turn. Each sample contains the full conversation history as the prompt, and the agent's response as the completion.

This benchmark replays those prompts against a target model and uses an **LLM-as-judge** to determine if the new response is functionally equivalent to the reference.

## Quick start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install anthropic pyyaml tenacity

# Dry run (no API calls, shows sample count + estimated tokens)
python run.py --config configs/eval-haiku.yaml --dry-run

# Run on a few samples
ANTHROPIC_API_KEY="..." python run.py --config configs/eval-haiku.yaml --max-samples 5

# Full run
ANTHROPIC_API_KEY="..." python run.py --config configs/eval-haiku.yaml
```

## How it works

```
For each sample:
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
run_name: eval-haiku-45

model:
  provider: anthropic
  model_name: claude-haiku-4-5-20251001
  temperature: 0.7
  max_tokens: 8192

judge:
  provider: anthropic
  model_name: claude-sonnet-4-5-20250929
  temperature: 0.0
  max_tokens: 2048

data_path: data/samples/all-solved.jsonl
filter_tasks: null       # e.g. ["dialogue-parser", "flood-risk-analysis"]
filter_turns: null        # e.g. [0] for first turn only
max_samples: null         # cap total samples

max_concurrent: 10
output_dir: results
max_retries: 2
```

## Results

Results are written to `results/{run_name}/`:

```
results/eval-haiku-45/
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
| `scripts/trajectory_to_samples.py` | Script to convert trajectories → JSONL samples |

### Tasks included (13 tasks, 181 samples)

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
