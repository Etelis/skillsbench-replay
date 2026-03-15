#!/usr/bin/env python3
"""
Convert agent trajectories from solved SkillsBench tasks into prompt/completion samples.

Each agent turn becomes one sample:
  - prompt:      full conversation history up to (and including) the latest user message
  - completion:  the agent's response at that turn
  - metadata:    task name, turn index, total turns, reward, model info

Usage:
    # Single trial
    python scripts/trajectory_to_samples.py \
        skillsbench-trajectories/jobs/haiku-45-solved/dialogue-parser__bwqCHfi \
        -o samples/dialogue-parser.jsonl

    # Entire job (all trials)
    python scripts/trajectory_to_samples.py \
        skillsbench-trajectories/jobs/haiku-45-solved \
        -o samples/all-solved.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_trajectory(trial_dir: Path) -> dict | None:
    traj_path = trial_dir / "agent" / "trajectory.json"
    if not traj_path.exists():
        return None
    with open(traj_path) as f:
        return json.load(f)


def load_reward(trial_dir: Path) -> float | None:
    result_path = trial_dir / "result.json"
    if result_path.exists():
        with open(result_path) as f:
            data = json.load(f)
            # Reward lives under verifier_result.rewards.reward
            rewards = (data.get("verifier_result") or {}).get("rewards") or {}
            reward = rewards.get("reward")
            if reward is not None:
                return reward
            # Fallback to top-level
            return data.get("reward")
    return None


def extract_task_name(trial_dir_name: str) -> str:
    """Extract task name from trial directory (e.g. 'dialogue-parser__bwqCHfi' -> 'dialogue-parser')."""
    parts = trial_dir_name.rsplit("__", 1)
    return parts[0] if len(parts) == 2 else trial_dir_name


def build_messages(steps: list[dict], up_to: int) -> list[dict]:
    """Build a chat-style message list from steps[0..up_to] (inclusive)."""
    messages = []
    for step in steps[: up_to + 1]:
        role = "user" if step["source"] == "user" else "assistant"
        content = step["message"] if isinstance(step["message"], str) else json.dumps(step["message"])
        messages.append({"role": role, "content": content})
    return messages


def trajectory_to_samples(trial_dir: Path) -> list[dict]:
    traj = load_trajectory(trial_dir)
    if traj is None:
        return []

    steps = traj["steps"]
    reward = load_reward(trial_dir)
    task_name = extract_task_name(trial_dir.name)
    agent_info = traj.get("agent", {})
    model_info = agent_info.get("model", {})

    agent_indices = [i for i, s in enumerate(steps) if s["source"] == "agent"]
    total_agent_turns = len(agent_indices)

    samples = []
    for turn_num, agent_idx in enumerate(agent_indices):
        # Prompt = everything before this agent turn (steps 0..agent_idx-1)
        prompt_messages = build_messages(steps, agent_idx - 1)

        # Completion = the agent's response
        agent_step = steps[agent_idx]
        completion = agent_step["message"] if isinstance(agent_step["message"], str) else json.dumps(agent_step["message"])

        sample = {
            "prompt": prompt_messages,
            "completion": completion,
            "metadata": {
                "task_name": task_name,
                "trial_id": trial_dir.name,
                "turn": turn_num,
                "total_turns": total_agent_turns,
                "reward": reward,
                "agent": agent_info.get("name"),
                "model_provider": model_info.get("provider"),
                "model_name": model_info.get("name"),
            },
        }
        samples.append(sample)

    return samples


def find_trial_dirs(path: Path) -> list[Path]:
    """Given a path, return trial directories. Handles both single trial and job directory."""
    traj = path / "agent" / "trajectory.json"
    if traj.exists():
        return [path]

    # Job directory — find all subdirs with trajectories
    trials = []
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and (entry / "agent" / "trajectory.json").exists():
            trials.append(entry)
    return trials


def main():
    parser = argparse.ArgumentParser(description="Convert trajectories to prompt/completion samples")
    parser.add_argument("path", type=Path, help="Path to a trial directory or job directory")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON (one sample per line, indented)")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: {args.path} does not exist", file=sys.stderr)
        sys.exit(1)

    trial_dirs = find_trial_dirs(args.path)
    if not trial_dirs:
        print(f"Error: no trajectories found in {args.path}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    with open(args.output, "w") as out:
        for trial_dir in trial_dirs:
            samples = trajectory_to_samples(trial_dir)
            for sample in samples:
                if args.pretty:
                    out.write(json.dumps(sample, indent=2) + "\n")
                else:
                    out.write(json.dumps(sample) + "\n")
            task = extract_task_name(trial_dir.name)
            print(f"  {task}: {len(samples)} samples")
            total_samples += len(samples)

    print(f"\nWrote {total_samples} samples to {args.output}")


if __name__ == "__main__":
    main()
