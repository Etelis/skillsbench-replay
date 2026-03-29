# Skillsbench-Replay Results

## Setup

### Models

| Model | HuggingFace ID | vLLM Image | TP | Memory | max-model-len |
|-------|---------------|------------|-----|--------|---------------|
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | `ghcr.io/llm-d/llm-d-cuda:v0.5.0` | 1 | 64Gi | 65536 |
| Llama 3.3 70B | `meta-llama/Llama-3.3-70B-Instruct` | `ghcr.io/llm-d/llm-d-cuda:v0.5.0` | 8 | 128Gi | 65536 |
| Llama 4 Maverick FP8 | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | `vllm/vllm-openai:v0.18.0` | 8 | 256Gi | 65536 |
| Granite 4 H small | `ibm-granite/granite-4.0-h-small` | `vllm/vllm-openai:v0.18.0` | 4 | 128Gi | 65536 |

### Spans Image

`ghcr.io/almogtavor/vllm-segmented-spans:2026-03-16-2eba5612` with:
- `VLLM_USE_V1=1`, `VLLM_V1_SPANS_ENABLED=True`
- `VLLM_V1_SPANS_TOKEN_PLUS=10`, `VLLM_V1_SPANS_TOKEN_CROSS=31`
- `--attention-backend=TRITON_ATTN --enforce-eager --block-size=16`

Spans/naive only supported on Llama 3.x models. Maverick and Granite crash with `EngineCore encountered an issue` on the spans fork.

### Benchmark Configuration

- **Samples**: 165 (12 tasks, from Haiku 4.5 trajectories with reward >= 0.8)
- **Decoding**: `temperature=0`, `top_p=1.0`, `seed=42`, `max_tokens=8192`
- **Concurrency**: 1 (sequential)
- **Middleware**: Llama 8B/70B baselines through middleware (`SPAN_MODE=full`); Maverick/Granite direct to vLLM
- **Judge**: `openai/gpt-oss-120b` via RITS

### Infrastructure

- OpenShift cluster: `pokprod001.ete14.res.ibm.com`
- Namespace: `llm-d-pic`
- Helm chart: `llm-d-modelservice v0.4.5`

---

## Task Evaluation

Scores each model's output against Haiku 4.5 reference trajectories. Measures how well the model performs the agent task (not whether spans preserves output).

**Judge criteria**: intent, commands, analysis, safety (binary PASS/FAIL per criterion).

| Model | Samples | Avg Latency | Intent | Commands | Analysis | Safety | Overall |
|-------|---------|-------------|--------|----------|----------|--------|---------|
| Llama 3.1 8B | 165/165 | 4.89s | 33.9% | 0.0% | 24.2% | 35.8% | **23.5%** |
| Llama 3.3 70B | 165/165 | 9.10s | 42.4% | 0.0% | 28.5% | 37.0% | **27.0%** |
| Llama 4 Maverick FP8 | 165/165 | 15.67s | 55.2% | 0.0% | 32.7% | 48.5% | **34.1%** |
| Granite 4 H small | 165/165 | 25.43s | 47.3% | 0.0% | 30.3% | 36.4% | **28.5%** |

**Note**: Commands scored 0% across all models due to a `NoneType` error in the judge's `MatchClosestOption` processor for that criterion.

For reference, prior results from the skillsbench-replay README (using Sonnet 4.6 as judge):

| Model | Overall | Intent | Commands | Analysis | Safety |
|-------|---------|--------|----------|----------|--------|
| Claude Sonnet 4.6 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| GPT-OSS 120B | 68.6% | 63.6% | 46.6% | 84.7% | 79.5% |
| Llama 3.3 70B | 63.4% | 58.6% | 37.0% | 80.1% | 77.9% |
| Llama 3.1 8B | 46.8% | 43.7% | 23.2% | 64.8% | 55.6% |

---

## Spans Comparison

Compares spans/naive outputs against the **baseline output** (same model, full recomputation). Measures whether the middleware degrades quality.

**Judge criteria**: `correctness_based_on_ground_truth` + `consistency` (unitxt catalog, reference-based).

### Llama 3.1 8B

| Mode | Correctness | Consistency | Overall |
|------|-------------|-------------|---------|
| spans | 73.6% | 61.7% | **67.7%** |
| naive | 70.6% | 58.5% | **64.5%** |

### Llama 3.3 70B

| Mode | Correctness | Consistency | Overall |
|------|-------------|-------------|---------|
| spans | 85.5% | 75.6% | **80.5%** |
| naive | 84.2% | 75.3% | **79.8%** |

### Observations

- Spans and naive produce nearly identical scores for both models (within ~3%).
- 70B shows higher fidelity to baseline (~80%) than 8B (~66%).
- The ~20-35% gap from 100% is expected: the spans vLLM fork uses a different attention backend (TRITON_ATTN) and different KV cache management, so even with deterministic decoding (`temp=0, seed=42`) the outputs diverge.
- Maverick and Granite cannot run spans/naive — the spans fork crashes on MoE/Mamba architectures.

---

## Reproduction

### Run baseline

```bash
# Deploy model
helm install vllm-itay-llama8b llm-d-modelservice \
  --repo https://llm-d-incubation.github.io/llm-d-modelservice/ \
  --version v0.4.5 -n llm-d-pic \
  -f helm/values-template.yaml \
  --set 'modelArtifacts.uri=hf://meta-llama/Llama-3.1-8B-Instruct' \
  --set 'modelArtifacts.name=meta-llama/Llama-3.1-8B-Instruct' \
  --set 'decode.containers[0].image=ghcr.io/llm-d/llm-d-cuda:v0.5.0' \
  --set 'decode.containers[0].args[4]=--max-model-len=65536' \
  --set 'extraObjects[0].spec.selector.llm-d\.ai/model=Llama-3.1-8B-Instruct' \
  --wait --timeout 30m

# Port-forward
oc port-forward svc/vllm-itay-llama8b 8000:8000 -n llm-d-pic &

# Start middleware (full recomputation)
SPAN_MODE=full BACKEND_URL=http://localhost:8000 \
  MODEL_DEFS_PATH=middleware/model_defs.yaml \
  uv run uvicorn middleware.spans_middleware:app --port 9000 &

# Run benchmark
uv run python run_benchmark.py \
  --endpoint http://localhost:9000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --label baseline --max-concurrent 1
```

### Run spans/naive

Same as baseline but:
1. Use spans image: `ghcr.io/almogtavor/vllm-segmented-spans:2026-03-16-2eba5612`
2. Add spans env vars to helm deploy
3. Change middleware mode: `SPAN_MODE=spans` or `SPAN_MODE=naive`
4. Change label: `--label spans` or `--label naive`

### Run judge

```bash
# Task evaluation (vs Haiku reference)
uv run python run_judge.py \
  --results results/bench-baseline-*.json \
  --judge-endpoint https://inference.example.com/gpt-oss-120b/v1 \
  --judge-model openai/gpt-oss-120b \
  --api-key YOUR_KEY

# Spans comparison (vs baseline)
uv run python run_judge.py \
  --results results/bench-spans-*.json results/bench-naive-*.json \
  --baseline results/bench-baseline-TIMESTAMP.json \
  --judge-endpoint https://inference.example.com/gpt-oss-120b/v1 \
  --judge-model openai/gpt-oss-120b \
  --api-key YOUR_KEY
```

---

## Result Files

| File | Description |
|------|-------------|
| `bench-baseline-20260328T201348Z.json` | Llama 3.1 8B baseline (165/165) |
| `bench-baseline-20260328T204901Z.json` | Llama 3.3 70B baseline (165/165) |
| `bench-baseline-20260328T220855Z.json` | Llama 4 Maverick FP8 baseline (165/165) |
| `bench-baseline-20260328T235122Z.json` | Granite 4 H small baseline (165/165) |
| `bench-spans-20260328T172517Z.json` | Llama 3.1 8B spans (165/165) |
| `bench-naive-20260328T175412Z.json` | Llama 3.1 8B naive (165/165) |
| `bench-spans-20260328T184636Z.json` | Llama 3.3 70B spans (165/165) |
| `bench-naive-20260328T193505Z.json` | Llama 3.3 70B naive (165/165) |
| `judge-20260329T003121Z.json` | Task evaluation (all 4 baselines) |
| `judge-comparison-20260329T053648Z.json` | Spans comparison (8B) |
| `judge-comparison-20260329T054023Z.json` | Spans comparison (70B) |
