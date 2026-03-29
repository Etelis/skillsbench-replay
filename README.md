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

## Source

Trajectories are extracted from the [SkillsBench](https://github.com/harbor-ai/skillsbench) benchmark using the `terminus-2-skills` agent with Haiku 4.5. The full trajectory (ATIF-v1.6 format) is used to capture all conversation turns including skill-loading exchanges.
