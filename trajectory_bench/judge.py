import json
import re
from dataclasses import dataclass

from .config import JudgeConfig
from .data import Sample
from .llm import judge_call

JUDGE_PROMPT = """You are evaluating whether two AI agent responses are functionally equivalent.

Context: An AI agent is solving a command-line task. Given the same conversation history,
two different models produced two responses. Determine if the candidate response would lead
to equivalent progress on the task as the reference response.

## Task context
Task: {task_name}
Turn: {turn} of {total_turns}
Reward achieved by reference trajectory: {reward}

## Last user message (what the agent is responding to):
{last_user_message}

## Reference response (from a successful trajectory):
{reference}

## Candidate response (being evaluated):
{candidate}

## Evaluation criteria:
1. INTENT: Do both responses attempt the same logical next step?
2. COMMANDS: Would the commands achieve similar outcomes? (syntax may differ)
3. ANALYSIS: Is the candidate's understanding of the task state comparable?
4. ERRORS: Does the candidate make any clear mistakes that would derail the task?

## Scoring:
- "equivalent": Candidate would lead to same or better task progress. Commands may differ syntactically but must be functionally equivalent.
- "partially_equivalent": Candidate is on the right track but misses some aspect (e.g., correct analysis but suboptimal commands).
- "not_equivalent": Candidate would lead to significantly worse task progress.

Respond ONLY with JSON:
{{"verdict": "equivalent" | "partially_equivalent" | "not_equivalent", "reasoning": "Brief explanation"}}"""

VERDICT_SCORES = {
    "equivalent": 1.0,
    "partially_equivalent": 0.5,
    "not_equivalent": 0.0,
}


@dataclass
class JudgeResult:
    verdict: str
    score: float
    reasoning: str
    usage: dict


def _build_judge_prompt(sample: Sample, reference: str, candidate: str) -> str:
    last_user_msgs = [m for m in sample.prompt if m["role"] == "user"]
    last_user = last_user_msgs[-1]["content"] if last_user_msgs else ""

    if len(last_user) > 4000:
        last_user = last_user[:2000] + "\n...[truncated]...\n" + last_user[-2000:]

    return JUDGE_PROMPT.format(
        task_name=sample.metadata["task_name"],
        turn=sample.metadata["turn"] + 1,
        total_turns=sample.metadata["total_turns"],
        reward=sample.metadata.get("reward", "N/A"),
        last_user_message=last_user,
        reference=reference,
        candidate=candidate,
    )


def _parse_verdict(text: str) -> tuple[str, str]:
    """Parse judge response, return (verdict, reasoning)."""
    # Try JSON parse
    try:
        # Find JSON in the response
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data["verdict"], data.get("reasoning", "")
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: look for verdict keyword
    text_lower = text.lower()
    for verdict in ["equivalent", "partially_equivalent", "not_equivalent"]:
        if verdict in text_lower:
            return verdict, text

    return "not_equivalent", f"Failed to parse judge response: {text[:200]}"


async def judge_compare(
    config: JudgeConfig, sample: Sample, reference: str, candidate: str
) -> JudgeResult:
    prompt = _build_judge_prompt(sample, reference, candidate)
    response_text, usage = await judge_call(config, prompt)

    verdict, reasoning = _parse_verdict(response_text)
    score = VERDICT_SCORES.get(verdict, 0.0)

    return JudgeResult(verdict=verdict, score=score, reasoning=reasoning, usage=usage)
