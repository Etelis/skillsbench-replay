# skillsbench-replay

Evaluate small and self-hosted models on [SkillsBench](https://github.com/harbor-ai/skillsbench) tasks — without running any tools or environments.

## Motivation

Running SkillsBench end-to-end requires Docker containers, tool execution, and environment setup for each task. This makes it impractical to evaluate smaller or self-hosted models.

**skillsbench-replay** takes a different approach: we first ran SkillsBench with Haiku 4.5 (itself a relatively weak model) and kept only the tasks it solved successfully. From those solved runs, we extracted the full agent trajectories — every prompt the model received and every response it produced.

To evaluate a new model, we **replay the prompts** against it and use an **LLM-as-judge** to check if the responses are functionally equivalent to the known-good ones. No tools, no Docker, no environment setup — just inference.

## Data

### Structure

```
benchmark/
  metadata.json                          # Top-level: task list, sample counts, selection criteria
  flood-risk-analysis/
    metadata.json                        # Task-level: reward, model, trial info
    sample_0/
      prompt.json                        # {"messages": [{"role": "user", "content": "..."}]}
      response.json                      # {"role": "assistant", "content": "..."}
    sample_1/
      prompt.json                        # Full conversation history + current prompt
      response.json                      # Haiku's ground truth response
    ...
```

Each sample is **self-contained**: `prompt.json` includes the full message history (all prior user/assistant turns) plus the current prompt. A model can be evaluated on any sample independently by sending `messages` to the API and comparing the output against `response.json`.

### Tasks (12 tasks, 165 samples)

All tasks were successfully solved by Haiku 4.5 on SkillsBench (reward >= 0.8).

| Task | Samples | Reward |
|------|---------|--------|
| protein-expression-analysis | 33 | 1.000 |
| hvac-control | 21 | 1.000 |
| weighted-gdp-calc | 20 | 1.000 |
| mario-coin-counting | 18 | 1.000 |
| virtualhome-agent-planning | 15 | 1.000 |
| econ-detrending-correlation | 14 | 1.000 |
| dialogue-parser | 12 | 0.833 |
| flood-risk-analysis | 8 | 1.000 |
| video-filler-word-remover | 7 | 1.000 |
| offer-letter-generator | 6 | 1.000 |
| threejs-structure-parser | 6 | 1.000 |
| gravitational-wave-detection | 5 | 1.000 |

### Sample format

**prompt.json** — the input to the model:

```json
{
  "messages": [
    {"role": "user", "content": "You are an AI assistant tasked with..."},
    {"role": "assistant", "content": "I'll start by loading the skill..."},
    {"role": "user", "content": "Loaded skill: nws-flood-thresholds\n---\n..."},
    {"role": "assistant", "content": "{\"analysis\": \"...\", \"commands\": [...]}"},
    {"role": "user", "content": "New Terminal Output:\n04031000\n04036000\n..."}
  ]
}
```

**response.json** — the ground truth:

```json
{
  "role": "assistant",
  "content": "{\"analysis\": \"...\", \"plan\": \"...\", \"commands\": [...]}"
}
```

## Evaluation methodology

The judge uses **per-criterion binary assessment** — each criterion is evaluated in a separate LLM call with a PASS/FAIL verdict. This follows established methodologies:

- **G-Eval** ([Liu et al., 2023](https://arxiv.org/abs/2303.16634)): One dimension per LLM call produces more consistent results than multi-dimensional simultaneous evaluation.
- **Autorubric** ([2025](https://arxiv.org/html/2603.00077)): Binary MET/UNMET criteria evaluated independently prevent criterion conflation.
- **"Rubric Is All You Need"** ([2025](https://arxiv.org/html/2503.23989v1)): Pointwise binary rubric evaluation for code.

### Criteria

| Criterion | PASS | FAIL |
|-----------|------|------|
| **Intent** | Both responses attempt the same logical next step | Candidate pursues a different goal or skips a step |
| **Commands** | Commands would produce equivalent results | Commands would produce materially different results |
| **Analysis** | Candidate correctly reads the current task state | Candidate misinterprets output, errors, or prior results |
| **Safety** | No mistakes that would derail task progress | Introduces errors that break the task or block future progress |

## Previous results

We evaluated six models against the Haiku 4.5 reference trajectories, using Claude Sonnet 4.6 as judge.

| Model | Overall | Intent | Commands | Analysis | Safety |
|-------|---------|--------|----------|----------|--------|
| Claude Sonnet 4.6 | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |
| GPT-OSS 120B | **68.6%** | 63.6% | 46.6% | 84.7% | 79.5% |
| Maverick 17B | **63.7%** | 61.3% | 36.5% | 85.1% | 71.8% |
| Llama 3.3 70B | **63.4%** | 58.6% | 37.0% | 80.1% | 77.9% |
| Granite 4.0 8B | **48.6%** | 42.5% | 24.3% | 54.1% | 73.5% |
| Llama 3.1 8B | **46.8%** | 43.7% | 23.2% | 64.8% | 55.6% |

## Source

Trajectories are extracted from the [SkillsBench](https://github.com/harbor-ai/skillsbench) benchmark using the `terminus-2-skills` agent with Haiku 4.5. The full trajectory (ATIF-v1.6 format) is used to capture all conversation turns including skill-loading exchanges.
