"""
agent.py — Core agent loop.

Flow:
  1. Build a planning prompt that lists available tools.
  2. Send it to Ollama — ask the LLM to return a JSON array of tool calls.
  3. Execute each tool call in order, collecting results.
  4. If the LLM included a RESULT: section, use it directly.
     Otherwise, send tool results back to Ollama for a final synthesis.

The `log` callback is called after every significant step so callers
can persist incremental progress (e.g. write to the DB in real time).
"""

from __future__ import annotations

import json
import re
from typing import Callable

import httpx

from .config import settings
from .tools import call_tool, tools_description


# ── Prompts ───────────────────────────────────────────────────────────────────

_PLANNER_TEMPLATE = """\
You are TaskForge, an AI automation agent. Your job is to complete tasks by \
calling tools in sequence.

AVAILABLE TOOLS:
{tools}

TASK: {prompt}

INSTRUCTIONS:
1. Think carefully about what steps are needed to complete the task.
2. Output a JSON array of tool calls. Each element must be an object with:
   - "tool"   : exact tool name (string)
   - "args"   : object of argument name → value
   - "reason" : one sentence explaining why this step is needed
3. After the JSON array, write "RESULT:" followed by your final answer \
(if you already know it), or leave it blank and results will be synthesised.
4. Use at most {max_calls} tool calls. Be efficient.
5. If no tools are needed, output an empty array [] and write the answer after RESULT:.

EXAMPLE OUTPUT FORMAT:
[
  {{"tool": "http_get", "args": {{"url": "https://example.com"}}, "reason": "Fetch the page content"}},
  {{"tool": "write_file", "args": {{"filename": "out.txt", "content": "..."}}, "reason": "Save result"}}
]
RESULT: The task is complete. I fetched the page and saved it to out.txt.
"""

_SYNTHESIS_TEMPLATE = """\
You completed a task by running the following steps.

ORIGINAL TASK: {prompt}

TOOL RESULTS:
{results}

Based on the above, write a clear, concise final answer to the task. \
Include relevant data from the tool results. Do not repeat the steps — \
just give the final answer.
"""


# ── LLM helper ────────────────────────────────────────────────────────────────


async def _call_ollama(prompt: str) -> str:
    """Send a single-turn message to Ollama and return the response text."""
    async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
        r = await client.post(
            f"{settings.ollama_endpoint}/v1/chat/completions",
            json={
                "model":       settings.ollama_model,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens":  1400,
                "stream":      False,
            },
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return content.strip()


# ── Plan parser ───────────────────────────────────────────────────────────────


def _parse_plan(text: str) -> tuple[list[dict], str]:
    """
    Extract (tool_calls, final_answer) from the LLM output.

    The LLM is asked to output:
      <JSON array>
      RESULT: <optional answer>

    Returns an empty list and the raw text if no valid JSON array is found.
    """
    final_answer = ""

    if "RESULT:" in text:
        before, _, after = text.partition("RESULT:")
        text         = before
        final_answer = after.strip()

    # Find the outermost [...] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return [], final_answer

    try:
        calls = json.loads(match.group())
        if isinstance(calls, list):
            return calls, final_answer
    except json.JSONDecodeError:
        pass

    return [], final_answer


# ── Main agent loop ───────────────────────────────────────────────────────────


async def run_agent(
    prompt: str,
    log: Callable[[str], None],
) -> str:
    """
    Plan and execute a task.

    Args:
        prompt: natural-language task description
        log:    callback called with a string after each significant step;
                used by the caller to persist incremental progress

    Returns:
        Final result string (always non-empty).
    """
    log(f"Task received: {prompt}")

    # ── Step 1: ask LLM to plan ───────────────────────────────────────────────
    planner_prompt = _PLANNER_TEMPLATE.format(
        tools     = tools_description(),
        prompt    = prompt,
        max_calls = settings.max_tool_calls,
    )
    log("Sending task to LLM for planning...")

    try:
        plan_text = await _call_ollama(planner_prompt)
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot reach Ollama at {settings.ollama_endpoint}. "
            "Make sure Ollama is running: ollama serve"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama returned HTTP {e.response.status_code}")

    log(f"Plan received ({len(plan_text)} chars)")

    tool_calls, final_answer = _parse_plan(plan_text)

    # ── Step 2: execute tool calls ────────────────────────────────────────────
    if not tool_calls:
        log("No tool calls — returning direct LLM answer")
        return final_answer or plan_text

    log(f"Executing {len(tool_calls)} tool call(s)...")
    tool_results: list[dict] = []

    for i, call in enumerate(tool_calls[: settings.max_tool_calls]):
        tool_name = call.get("tool", "")
        tool_args = call.get("args", {})
        reason    = call.get("reason", "")

        # Format args for the log (truncate long values)
        args_preview = ", ".join(
            f"{k}={repr(v)[:50]}" for k, v in tool_args.items()
        )
        log(f"[{i + 1}/{len(tool_calls)}] {tool_name}({args_preview}) — {reason}")

        result = await call_tool(tool_name, tool_args)

        preview = result[:150] + ("..." if len(result) > 150 else "")
        log(f"    Result: {preview}")

        tool_results.append({"tool": tool_name, "args": tool_args, "result": result})

    # ── Step 3: return answer ─────────────────────────────────────────────────
    if final_answer:
        log("Using LLM-provided final answer")
        return final_answer

    # LLM didn't include a RESULT: — synthesise from tool outputs
    log("Synthesising final answer from tool results...")
    results_block = "\n".join(
        f"[{r['tool']}]: {r['result']}" for r in tool_results
    )
    synthesis_prompt = _SYNTHESIS_TEMPLATE.format(
        prompt  = prompt,
        results = results_block,
    )
    final = await _call_ollama(synthesis_prompt)
    log("Done.")
    return final
