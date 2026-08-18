"""The metalevel rollout: repeatedly call the LLM, execute whatever tool
calls it makes, and feed results back -- until it stops issuing tool calls
(a final report) or a hard computational-step safety cap is hit.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from llm.base import LLMClient, Message
from .tools import build_tools

MAX_STEPS = 400  # safety cap on computational actions; the evaluation budget is self-regulated via the prompt
BUDGET_NUDGE_FACTOR = 1.5

VERBOSE = os.environ.get("SARA_VERBOSE", "0") != "0"


USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens")


@dataclass
class RunResult:
    final_message: str
    n_steps: int
    transcript: list[dict] = field(default_factory=list)
    # Summed across every LLM turn in the campaign; None if the provider never reported
    # usage for a single turn (rather than a misleading all-zero total).
    usage_total: dict | None = None


def _count_observed(sandbox: Path) -> int | None:
    state_path = sandbox / "state.json"
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text())
        return sum(1 for t in state.get("trials", []) if t.get("status") == "observed")
    except (json.JSONDecodeError, OSError):
        return None


def _progress_line(sandbox: Path, budget: int | None) -> str | None:
    """One line per real evaluation, printed regardless of VERBOSE -- mirrors
    what a plain lenz-only loop (e.g. examples/branin/run_manual.py) already
    prints by default, so `sara run` isn't silent-until-the-end by comparison.
    """
    state_path = sandbox / "state.json"
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    objectives = state.get("shelf", {}).get("objectives", [])
    trials = [t for t in state.get("trials", []) if t.get("status") == "observed" and t.get("metrics")]
    if not trials:
        return None
    trials.sort(key=lambda t: t.get("observed_at") or 0)
    latest = trials[-1]
    n = len(trials)
    count = f"{n}/{budget}" if budget else str(n)

    if len(objectives) == 1:
        metric, minimize = objectives[0]["metric"], objectives[0]["minimize"]
        values = [t["metrics"][metric] for t in trials if metric in t.get("metrics", {})]
        if metric not in latest["metrics"] or not values:
            return f"eval {count}"
        best = min(values) if minimize else max(values)
        return f"eval {count}  {metric}={latest['metrics'][metric]:.4g}  best={best:.4g}"
    return f"eval {count}  ({len(objectives)} objectives)"


def run_campaign(
    client: LLMClient,
    system_prompt: str,
    user_prompt: str,
    sandbox: Path,
    trace_path: Path | None = None,
    budget: int | None = None,
    max_steps: int = MAX_STEPS,
) -> RunResult:
    tools, handlers = build_tools(sandbox)
    history: list[Message] = [Message(role="user", content=user_prompt)]
    transcript: list[dict] = []
    nudged = False
    usage_total = {f: 0 for f in USAGE_FIELDS}
    usage_seen = False

    if trace_path is not None:
        trace_path.write_text("")  # fresh trace per run, not appended to a stale file from a previous run

    def log(entry: dict) -> None:
        entry = {"ts": time.time(), **entry}
        transcript.append(entry)
        if trace_path is not None:
            with open(trace_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    log({"role": "user", "content": user_prompt})
    last_progress_count = 0

    for step in range(max_steps):
        if VERBOSE:
            n_observed = _count_observed(sandbox)
            print(f"[agent] step {step + 1}/{max_steps} (n_observed={n_observed})", file=sys.stderr, flush=True)

        response = client.chat(history, tools, system_prompt)
        history.append(Message(role="assistant", content=response.content, tool_calls=response.tool_calls))
        log(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls],
                "usage": response.usage,
            }
        )
        if response.usage:
            usage_seen = True
            for f in USAGE_FIELDS:
                v = response.usage.get(f)
                if v is not None:
                    usage_total[f] += v

        if not response.tool_calls:
            if VERBOSE:
                print(f"[agent] final report after {step + 1} steps", file=sys.stderr, flush=True)
            return RunResult(
                final_message=response.content,
                n_steps=step + 1,
                transcript=transcript,
                usage_total=usage_total if usage_seen else None,
            )

        for tc in response.tool_calls:
            handler = handlers.get(tc.name)
            t_tool = time.monotonic()
            result = handler(**tc.arguments) if handler else f"[error] unknown tool '{tc.name}'"
            if VERBOSE:
                elapsed = time.monotonic() - t_tool
                preview = result if len(result) <= 200 else result[:200] + "..."
                print(f"[agent]    tool={tc.name} args={tc.arguments} ({elapsed:.2f}s) -> {preview}", file=sys.stderr, flush=True)
            history.append(Message(role="tool", tool_call_id=tc.id, tool_name=tc.name, content=result))
            log({"role": "tool", "tool_name": tc.name, "arguments": tc.arguments, "content": result})

        n_observed = _count_observed(sandbox)
        if n_observed is not None and n_observed > last_progress_count:
            last_progress_count = n_observed
            line = _progress_line(sandbox, budget)
            if line:
                print(line, file=sys.stderr, flush=True)

        if budget and not nudged:
            if n_observed is not None and n_observed > budget * BUDGET_NUDGE_FACTOR:
                nudged = True
                nudge = (
                    f"You have used {n_observed} evaluations against a budget of {budget}. "
                    "Wrap up now: report the best feasible incumbent (or Pareto front) and stop."
                )
                history.append(Message(role="user", content=nudge))
                log({"role": "user", "content": nudge})

    return RunResult(
        final_message="[stopped: max computational steps reached without a final report]",
        n_steps=max_steps,
        transcript=transcript,
        usage_total=usage_total if usage_seen else None,
    )
