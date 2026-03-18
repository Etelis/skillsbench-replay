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

We evaluated six models against the Haiku 4.5 reference trajectories, using Claude Sonnet 4.6 as judge.

#### Overall

| Model | Overall | Intent | Commands | Analysis | Safety |
|-------|---------|--------|----------|----------|--------|
| Claude Sonnet 4.6 | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |
| GPT-OSS 120B | **68.6%** | 63.6% | 46.6% | 84.7% | 79.5% |
| Maverick 17B | **63.7%** | 61.3% | 36.5% | 85.1% | 71.8% |
| Llama 3.3 70B | **63.4%** | 58.6% | 37.0% | 80.1% | 77.9% |
| Granite 4.0 8B | **48.6%** | 42.5% | 24.3% | 54.1% | 73.5% |
| Llama 3.1 8B | **46.8%** | 43.7% | 23.2% | 64.8% | 55.6% |

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

#### Per-task breakdown (Maverick 17B)

| Task | Score | Intent | Commands | Analysis | Safety | N |
|------|-------|--------|----------|----------|--------|---|
| flood-risk-analysis | 81.2% | 75% | 62% | 100% | 88% | 8 |
| offer-letter-generator | 79.2% | 83% | 67% | 100% | 67% | 6 |
| gravitational-wave-detection | 70.0% | 60% | 40% | 100% | 80% | 5 |
| virtualhome-agent-planning | 70.0% | 60% | 53% | 93% | 73% | 15 |
| dialogue-parser | 68.8% | 58% | 42% | 100% | 75% | 12 |
| mario-coin-counting | 68.1% | 61% | 39% | 83% | 89% | 18 |
| protein-expression-analysis | 65.9% | 70% | 30% | 94% | 70% | 33 |
| threejs-structure-parser | 62.5% | 50% | 50% | 83% | 67% | 6 |
| weighted-gdp-calc | 61.3% | 70% | 35% | 70% | 70% | 20 |
| video-filler-word-remover | 60.7% | 71% | 29% | 86% | 57% | 7 |
| hvac-control | 58.3% | 48% | 19% | 86% | 81% | 21 |
| econ-detrending-correlation | 57.1% | 57% | 43% | 79% | 50% | 14 |
| lab-unit-harmonization | 45.3% | 44% | 19% | 56% | 62% | 16 |

#### Per-task breakdown (Granite 4.0 8B)

| Task | Score | Intent | Commands | Analysis | Safety | N |
|------|-------|--------|----------|----------|--------|---|
| dialogue-parser | 68.8% | 50% | 58% | 83% | 83% | 12 |
| threejs-structure-parser | 66.7% | 67% | 33% | 83% | 83% | 6 |
| offer-letter-generator | 62.5% | 83% | 33% | 67% | 67% | 6 |
| video-filler-word-remover | 53.6% | 43% | 43% | 57% | 71% | 7 |
| weighted-gdp-calc | 52.5% | 60% | 20% | 65% | 65% | 20 |
| protein-expression-analysis | 51.5% | 52% | 24% | 61% | 70% | 33 |
| hvac-control | 50.0% | 38% | 19% | 52% | 90% | 21 |
| gravitational-wave-detection | 45.0% | 20% | 40% | 60% | 60% | 5 |
| lab-unit-harmonization | 43.8% | 38% | 12% | 44% | 81% | 16 |
| econ-detrending-correlation | 42.9% | 36% | 29% | 50% | 57% | 14 |
| flood-risk-analysis | 37.5% | 38% | 25% | 38% | 50% | 8 |
| mario-coin-counting | 37.5% | 28% | 11% | 39% | 72% | 18 |
| virtualhome-agent-planning | 35.0% | 13% | 13% | 27% | 87% | 15 |

#### Per-task breakdown (Llama 3.1 8B)

| Task | Score | Intent | Commands | Analysis | Safety | N |
|------|-------|--------|----------|----------|--------|---|
| dialogue-parser | 60.4% | 58% | 25% | 83% | 75% | 12 |
| mario-coin-counting | 56.9% | 50% | 33% | 89% | 56% | 18 |
| threejs-structure-parser | 54.2% | 17% | 33% | 83% | 83% | 6 |
| protein-expression-analysis | 50.0% | 64% | 18% | 64% | 55% | 22 |
| virtualhome-agent-planning | 48.3% | 47% | 40% | 53% | 53% | 15 |
| econ-detrending-correlation | 46.4% | 43% | 29% | 64% | 50% | 14 |
| weighted-gdp-calc | 45.0% | 40% | 20% | 60% | 60% | 10 |
| flood-risk-analysis | 43.8% | 38% | 38% | 62% | 38% | 8 |
| video-filler-word-remover | 42.9% | 43% | 0% | 100% | 29% | 7 |
| offer-letter-generator | 41.7% | 33% | 33% | 50% | 50% | 6 |
| lab-unit-harmonization | 33.3% | 44% | 0% | 22% | 67% | 9 |
| gravitational-wave-detection | 30.0% | 20% | 20% | 40% | 40% | 5 |
| hvac-control | 30.0% | 10% | 0% | 50% | 60% | 10 |

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
| `data/rounds/cache-manifest.json` | Dataset-level caching metadata |
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

### Cache hints metadata

Each round includes `cache_hints` in its metadata, marking which message blocks are cacheable and how. This is useful for KV cache benchmarking — particularly for evaluating prefix caching and position-independent caching (PIC) implementations.

#### Per-round annotations

Each round's `metadata.cache_hints` is an array of cacheable blocks:

```json
{
  "cache_hints": [
    {
      "message_index": 0,
      "type": "prefix",
      "block": "system_prompt",
      "hash": "4cb42d9c00fc46be",
      "char_range": [0, 5192],
      "shared_prefix_chars": 3036
    },
    {
      "message_index": 2,
      "type": "pic",
      "block": "skill_definition",
      "name": "xlsx",
      "hash": "adb04f04a2378276",
      "char_len": 4062,
      "reusable_across_tasks": true
    }
  ]
}
```

#### Cache types

| Type | Meaning | Example |
|------|---------|---------|
| `prefix` | Identical content always at the same position — standard prefix caching | System prompts: 3,036 chars shared across all 13 tasks, then task-specific suffixes (4,149–6,196 chars total) |
| `pic` | Identical content appearing at different absolute positions — requires position-independent caching | Skill definitions loaded mid-conversation: same `xlsx` skill (4,062 chars) used in both `protein-expression-analysis` and `weighted-gdp-calc` at different offsets |
