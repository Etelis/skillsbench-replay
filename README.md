# skillsbench-replay

Evaluate small and self-hosted models on [SkillsBench](https://github.com/harbor-ai/skillsbench) tasks — without running any tools or environments.

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

### Results

We evaluated three models against the Haiku 4.5 reference trajectories, using Claude Sonnet 4.6 as judge.

#### Overall

| Model | Overall | Intent | Commands | Analysis | Safety |
|-------|---------|--------|----------|----------|--------|
| Claude Sonnet 4.6 | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |
| GPT-OSS 120B | **68.6%** | 63.6% | 46.6% | 84.7% | 79.5% |
| Llama 3.3 70B | **63.4%** | 58.6% | 37.0% | 80.1% | 77.9% |

#### Per-task breakdown (Claude Sonnet 4.6)

| Task | Score | Intent | Commands | Analysis | Safety | N |
|------|-------|--------|----------|----------|--------|---|
| dialogue-parser | 100.0% | 100% | 100% | 100% | 100% | 12 |
| econ-detrending-correlation | 100.0% | 100% | 100% | 100% | 100% | 14 |
| flood-risk-analysis | 100.0% | 100% | 100% | 100% | 100% | 8 |
| gravitational-wave-detection | 100.0% | 100% | 100% | 100% | 100% | 5 |
| hvac-control | 100.0% | 100% | 100% | 100% | 100% | 21 |
| lab-unit-harmonization | 100.0% | 100% | 100% | 100% | 100% | 16 |
| mario-coin-counting | 100.0% | 100% | 100% | 100% | 100% | 18 |
| offer-letter-generator | 100.0% | 100% | 100% | 100% | 100% | 6 |
| protein-expression-analysis | 100.0% | 100% | 100% | 100% | 100% | 33 |
| threejs-structure-parser | 100.0% | 100% | 100% | 100% | 100% | 6 |
| video-filler-word-remover | 100.0% | 100% | 100% | 100% | 100% | 7 |
| virtualhome-agent-planning | 100.0% | 100% | 100% | 100% | 100% | 15 |
| weighted-gdp-calc | 100.0% | 100% | 100% | 100% | 100% | 20 |

#### Per-task breakdown (GPT-OSS 120B)

| Task | Score | Intent | Commands | Analysis | Safety | N |
|------|-------|--------|----------|----------|--------|---|
| offer-letter-generator | 91.7% | 83% | 83% | 100% | 100% | 6 |
| gravitational-wave-detection | 85.0% | 80% | 60% | 100% | 100% | 5 |
| dialogue-parser | 81.2% | 50% | 75% | 100% | 100% | 12 |
| flood-risk-analysis | 81.2% | 62% | 75% | 88% | 100% | 8 |
| econ-detrending-correlation | 78.6% | 79% | 57% | 93% | 86% | 14 |
| virtualhome-agent-planning | 76.7% | 60% | 53% | 100% | 93% | 15 |
| mario-coin-counting | 69.4% | 67% | 50% | 94% | 67% | 18 |
| weighted-gdp-calc | 68.8% | 70% | 50% | 85% | 70% | 20 |
| threejs-structure-parser | 66.7% | 33% | 50% | 100% | 83% | 6 |
| lab-unit-harmonization | 64.1% | 62% | 25% | 81% | 88% | 16 |
| video-filler-word-remover | 60.7% | 57% | 43% | 86% | 57% | 7 |
| protein-expression-analysis | 59.1% | 67% | 36% | 67% | 67% | 33 |
| hvac-control | 50.0% | 50% | 12% | 62% | 75% | 16 |

#### Per-task breakdown (Llama 3.3 70B)

| Task | Score | Intent | Commands | Analysis | Safety | N |
|------|-------|--------|----------|----------|--------|---|
| gravitational-wave-detection | 85.0% | 80% | 80% | 100% | 80% | 5 |
| threejs-structure-parser | 79.2% | 67% | 67% | 100% | 83% | 6 |
| dialogue-parser | 77.1% | 58% | 67% | 100% | 83% | 12 |
| offer-letter-generator | 66.7% | 67% | 50% | 67% | 83% | 6 |
| virtualhome-agent-planning | 66.7% | 47% | 53% | 87% | 80% | 15 |
| flood-risk-analysis | 65.6% | 50% | 38% | 88% | 88% | 8 |
| mario-coin-counting | 65.3% | 56% | 33% | 83% | 89% | 18 |
| video-filler-word-remover | 64.3% | 71% | 29% | 100% | 57% | 7 |
| weighted-gdp-calc | 63.7% | 70% | 35% | 75% | 75% | 20 |
| protein-expression-analysis | 62.9% | 70% | 30% | 82% | 70% | 33 |
| econ-detrending-correlation | 60.7% | 64% | 43% | 71% | 64% | 14 |
| lab-unit-harmonization | 57.8% | 56% | 19% | 62% | 94% | 16 |
| hvac-control | 46.4% | 29% | 14% | 67% | 76% | 21 |

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
