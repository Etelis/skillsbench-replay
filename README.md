# skillsbench-replay

Evaluate small and self-hosted models on [SkillsBench](https://github.com/harbor-ai/skillsbench) tasks — without running any tools or environments. Designed for models served via **vLLM**.

## Motivation

Running SkillsBench end-to-end requires Docker containers, tool execution, and environment setup for each task. This makes it impractical to evaluate smaller or self-hosted models.

**skillsbench-replay** takes a different approach: we first ran SkillsBench with Haiku 4.5 (itself a relatively weak model) and kept only the tasks it solved successfully. From those solved runs, we extracted the full agent trajectories — every prompt the model received and every response it produced.

To evaluate a new model, we **replay the prompts** against it and use an **LLM-as-judge** to check if the responses are functionally equivalent to the known-good ones. No tools, no Docker, no environment setup — just inference. This makes it possible to benchmark models served locally via vLLM on consumer hardware.

## Quick start

### 1. Serve your model with vLLM

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

### 2. Install and run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install openai pyyaml tenacity

# Dry run — shows round count + estimated tokens
python run.py --endpoint http://localhost:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct --dry-run

# Test with a few rounds
python run.py --endpoint http://localhost:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct --max-rounds 5

# Full run (181 rounds)
python run.py --endpoint http://localhost:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct
```

### With kvcache-bench

This benchmark integrates with [kvcache-bench](https://github.ibm.com/AICoOptimization/kvcache-bench). When registered there, run via:

```bash
python3 -m src llama8b --benchmarks skillsbench-replay
```

Or locally with port-forward:

```bash
oc port-forward -n llm-d-pic svc/llm-d-inference 8000:8000 &
python benchmarks/skillsbench-replay/run.py \
  --endpoint http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

## How it works

```
For each of the 181 rounds:
  1. Send the prompt (conversation history) to the eval model (vLLM)
  2. Get the candidate completion
  3. Judge evaluates candidate vs reference on 4 criteria (one LLM call each)
  4. Score = passed_criteria / 4  (0.0, 0.25, 0.5, 0.75, 1.0)
```

### Evaluation methodology

The judge uses **per-criterion binary assessment** — each criterion is evaluated in a separate LLM call with a PASS/FAIL verdict. This follows established methodologies:

- **G-Eval** ([Liu et al., 2023](https://arxiv.org/abs/2303.16634)): One dimension per LLM call produces more consistent results than multi-dimensional simultaneous evaluation. Chain-of-thought reasoning before verdict improves accuracy.
- **Autorubric** ([2025](https://arxiv.org/html/2603.00077)): Binary MET/UNMET criteria evaluated independently prevent criterion conflation, where strength in one dimension inflates others.
- **"Rubric Is All You Need"** ([2025](https://arxiv.org/html/2503.23989v1)): Pointwise binary rubric evaluation for code — one criterion per call with YES/NO decision.

### Criteria

Each criterion is evaluated independently with its own prompt:

| Criterion | PASS | FAIL |
|-----------|------|------|
| **Intent** | Both responses attempt the same logical next step | Candidate pursues a different goal or skips a step |
| **Commands** | Commands would produce equivalent results | Commands would produce materially different results |
| **Analysis** | Candidate correctly reads the current task state | Candidate misinterprets output, errors, or prior results |
| **Safety** | No mistakes that would derail task progress | Introduces errors that break the task or block future progress |

Each judge call follows the pattern: task context → reference response → candidate response → criterion definition with PASS/FAIL descriptions → "explain your reasoning step by step, then provide your verdict".

### Validation

We validated the benchmark by running Claude Sonnet 4.6 (a stronger model than the Haiku 4.5 that produced the reference trajectories) as both eval model and judge. It scored **100% across all 4 criteria on all tested rounds**, confirming the judge correctly identifies functionally equivalent responses and that a capable model can reliably match the reference behavior.

## Usage

### CLI args (kvcache-bench compatible)

```bash
python run.py \
  --endpoint http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max-rounds 20
```

Use a separate judge model/endpoint:

```bash
python run.py \
  --endpoint http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --judge-endpoint http://localhost:8001/v1 \
  --judge-model meta-llama/Llama-3.1-70B-Instruct
```

### YAML config (alternative)

```bash
python run.py --config configs/eval-vllm.yaml
```

```yaml
# configs/eval-vllm.yaml
run_name: eval-vllm-llama-8b

model:
  provider: openai
  model_name: meta-llama/Llama-3.1-8B-Instruct
  base_url: http://localhost:8000/v1
  temperature: 0.7
  max_tokens: 8192

judge:
  provider: openai
  model_name: meta-llama/Llama-3.1-8B-Instruct
  base_url: http://localhost:8000/v1
  temperature: 0.0
  max_tokens: 2048

data_path: data/rounds/all-solved.jsonl
max_concurrent: 10
output_dir: results
```

### Filtering options

```yaml
filter_tasks: ["dialogue-parser", "flood-risk-analysis"]  # specific tasks only
filter_turns: [0]                                          # first turn only
max_rounds: 20                                            # cap total rounds
```

## Results

Results are written to `results/{run_name}/`:

```
results/eval-llama-3.1-8b-instruct/
├── config.yaml                          # Config snapshot
├── summary.json                         # Aggregate scores (overall, by-task, by-turn)
└── rounds/
    └── {task_name}/
        └── {round_id}.json             # Per-round: candidate, verdict, reasoning
```

Runs are **resumable** — restarting with the same config skips already-completed rounds.

## Data

| Directory | Contents |
|-----------|----------|
| `data/rounds/all-solved.jsonl` | 181 prompt/completion rounds (8.5MB) |
| `data/raw-trajectories/` | Original agent trajectories from 13 solved tasks |
| `scripts/trajectory_to_samples.py` | Script to convert trajectories into JSONL rounds |

### Tasks included (13 tasks, 181 rounds)

All tasks were successfully solved by Haiku 4.5 on SkillsBench.

| Task | Rounds | Reward |
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
