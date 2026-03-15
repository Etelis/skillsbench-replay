"""
LLM-as-judge evaluation using per-criterion binary assessment.

Methodology based on:
- G-Eval (Liu et al., 2023): One dimension per LLM call produces more
  consistent results than multi-dimensional simultaneous evaluation.
  Chain-of-thought reasoning before verdict improves accuracy.
  https://arxiv.org/abs/2303.16634

- Autorubric (2025): Binary MET/UNMET criteria evaluated independently
  via separate LLM calls. Prevents criterion conflation where strength
  in one dimension inflates others.
  https://arxiv.org/html/2603.00077

- "Rubric Is All You Need" (2025): Binary pointwise rubric evaluation
  (PRE) for code — one criterion per call with YES/NO decision.
  https://arxiv.org/html/2503.23989v1

Each sample is evaluated on 4 independent criteria, each in a separate
LLM call. The final score is passed_criteria / total_criteria.
"""

import json
import re
from dataclasses import dataclass

from .config import JudgeConfig
from .data import Sample
from .llm import judge_call

CRITERIA = {
    "intent": {
        "name": "Intent",
        "definition": (
            "Whether both responses attempt the same logical next step in solving the task. "
            "They should share the same goal for this turn — e.g., both explore a file, "
            "both install a dependency, both write the same kind of script."
        ),
        "pass": (
            "Both responses aim to accomplish the same thing this turn, even if "
            "the specific approach differs slightly."
        ),
        "fail": (
            "The candidate pursues a fundamentally different goal or skips/reorders "
            "a step that the reference considers necessary at this point."
        ),
    },
    "commands": {
        "name": "Commands",
        "definition": (
            "Whether the commands or code in the candidate response would produce "
            "equivalent outcomes to the reference. Syntax and style may differ — "
            "what matters is the end result on the filesystem, environment, or task state."
        ),
        "pass": (
            "The commands would produce the same or equivalent results. Minor differences "
            "in flags, variable names, or ordering are acceptable if the effect is the same."
        ),
        "fail": (
            "The commands would produce materially different results — wrong files, "
            "missing steps, incorrect arguments, or broken logic."
        ),
    },
    "analysis": {
        "name": "Analysis",
        "definition": (
            "Whether the candidate demonstrates a comparable understanding of the "
            "current task state. This includes correctly interpreting terminal output, "
            "error messages, file contents, or prior results."
        ),
        "pass": (
            "The candidate correctly reads the situation and identifies what has been "
            "done and what remains, even if phrased differently from the reference."
        ),
        "fail": (
            "The candidate misinterprets the current state — e.g., thinks a step "
            "succeeded when it failed, misreads output, or ignores important information."
        ),
    },
    "safety": {
        "name": "Safety",
        "definition": (
            "Whether the candidate avoids mistakes that would derail task progress. "
            "This includes destructive commands, incorrect assumptions that compound, "
            "or actions that would leave the environment in an unrecoverable state."
        ),
        "pass": (
            "The candidate does not introduce errors that would break the task or "
            "make future progress significantly harder."
        ),
        "fail": (
            "The candidate introduces a clear mistake — e.g., overwrites needed files, "
            "uses wrong APIs, makes an assumption that will cause failures downstream."
        ),
    },
}

CRITERION_PROMPT = """You are an impartial evaluator assessing an AI agent's response against a specific criterion.

## Task context
An AI agent is solving a command-line task called "{task_name}".
This is turn {turn} of {total_turns} in the trajectory.

## What the agent is responding to:
{last_user_message}

## Reference response (from a successful trajectory):
{reference}

## Candidate response (being evaluated):
{candidate}

## Criterion: {criterion_name}
{criterion_definition}

* PASS: {pass_description}
* FAIL: {fail_description}

First, explain your reasoning step by step. Then provide your verdict.

Respond in JSON:
{{"reasoning": "<your step-by-step analysis>", "verdict": "PASS" or "FAIL"}}"""


@dataclass
class CriterionResult:
    criterion: str
    verdict: str  # "PASS" or "FAIL"
    passed: bool
    reasoning: str


@dataclass
class JudgeResult:
    criteria: dict[str, CriterionResult]
    score: float  # passed_criteria / total_criteria
    usage: dict


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def _build_criterion_prompt(
    sample: Sample, reference: str, candidate: str, criterion_key: str
) -> str:
    criterion = CRITERIA[criterion_key]
    last_user_msgs = [m for m in sample.prompt if m["role"] == "user"]
    last_user = last_user_msgs[-1]["content"] if last_user_msgs else ""

    return CRITERION_PROMPT.format(
        task_name=sample.metadata["task_name"],
        turn=sample.metadata["turn"] + 1,
        total_turns=sample.metadata["total_turns"],
        last_user_message=_truncate(last_user),
        reference=_truncate(reference),
        candidate=_truncate(candidate),
        criterion_name=criterion["name"],
        criterion_definition=criterion["definition"],
        pass_description=criterion["pass"],
        fail_description=criterion["fail"],
    )


def _parse_criterion_verdict(text: str) -> tuple[str, str]:
    """Parse judge response, return (verdict, reasoning)."""
    try:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            verdict = data.get("verdict", "").upper().strip()
            if verdict in ("PASS", "FAIL"):
                return verdict, data.get("reasoning", "")
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: look for PASS/FAIL keywords
    text_upper = text.upper()
    if "PASS" in text_upper and "FAIL" not in text_upper:
        return "PASS", text
    if "FAIL" in text_upper:
        return "FAIL", text

    return "FAIL", f"Could not parse verdict: {text[:200]}"


async def judge_compare(
    config: JudgeConfig, sample: Sample, reference: str, candidate: str
) -> JudgeResult:
    """Evaluate candidate against reference on all criteria independently."""
    criteria_results: dict[str, CriterionResult] = {}
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    for criterion_key in CRITERIA:
        prompt = _build_criterion_prompt(sample, reference, candidate, criterion_key)
        response_text, usage = await judge_call(config, prompt)

        verdict, reasoning = _parse_criterion_verdict(response_text)
        passed = verdict == "PASS"

        criteria_results[criterion_key] = CriterionResult(
            criterion=criterion_key,
            verdict=verdict,
            passed=passed,
            reasoning=reasoning,
        )

        total_usage["input_tokens"] += usage.get("input_tokens", 0)
        total_usage["output_tokens"] += usage.get("output_tokens", 0)

    passed_count = sum(1 for r in criteria_results.values() if r.passed)
    score = passed_count / len(CRITERIA)

    return JudgeResult(criteria=criteria_results, score=score, usage=total_usage)
