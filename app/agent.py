"""
agent.py — Core agent loop.

Research-paper features implemented here:
  1. Plan repair      — retry LLM once if it returns malformed JSON
                        (Reflect, Retry, Reward — arxiv 2505.24726)
  2. ReAct thoughts   — explicit "thought" field before each tool call
                        (ReAct: Synergizing Reasoning and Acting, Yao et al.)
  3. Self-reflection  — agent verifies its own output after synthesis
                        (Reflexion / MAR — arxiv 2512.20845)
  4. TF-IDF tool pick — only send the K most relevant tools to the LLM
                        (Online-Optimised RAG for Tool Use — arxiv 2509.20415)
  5. Task memory      — caller can inject similar past results for context
                        (A-Mem: Agentic Memory for LLM Agents — arxiv 2502.12110)

Flow:
  1. Score tools against the prompt (TF-IDF), send only the top-K.
  2. Build planning prompt; optionally prepend task-memory context.
  3. Send to Ollama — expect a JSON array with thought+tool+args+reason.
  4. If JSON is malformed, repair-prompt the LLM once and retry.
  5. Execute each tool call in order, collecting results.
  6. Synthesise a final answer.
  7. Self-reflect: ask the LLM whether the answer is complete; apply any
     correction if it spots a problem.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Callable

import httpx

from .config import settings
from .tools import _REGISTRY, call_tool, tools_description


# ── Prompts ───────────────────────────────────────────────────────────────────

_PLANNER_TEMPLATE = """\
You are TaskForge, an AI automation agent. Your job is to complete tasks by \
calling tools in sequence.

AVAILABLE TOOLS:
{tools}

{memory_block}\
TASK: {prompt}

INSTRUCTIONS:
1. Think step by step about what is needed.
2. Output a JSON array of tool calls. Each element MUST be:
   {{
     "thought": "<one sentence of reasoning before deciding this action>",
     "tool"   : "<exact tool name>",
     "args"   : {{ "<arg>": "<value>", ... }},
     "reason" : "<one sentence explaining why this step is needed>"
   }}
3. After the JSON array write "RESULT:" and your final answer \
(if already known), or leave blank for synthesis.
4. Use at most {max_calls} tool calls.
5. If no tools are needed output [] and write the answer after RESULT:.

EXAMPLE:
[
  {{
    "thought": "I need to fetch the page before I can summarise it.",
    "tool": "http_get",
    "args": {{"url": "https://example.com"}},
    "reason": "Fetch page content"
  }}
]
RESULT:
"""

_REPAIR_TEMPLATE = """\
Your previous response could not be parsed as a valid JSON array.

Parse error: {error}

Your previous response (first 500 chars):
{bad_response}

Output ONLY a valid JSON array (no prose before it), then RESULT: and your answer.
The array must follow this schema exactly:
[{{"thought": "...", "tool": "...", "args": {{}}, "reason": "..."}}]
"""

_SYNTHESIS_TEMPLATE = """\
You completed a task by running the following tool calls.

ORIGINAL TASK: {prompt}

TOOL RESULTS:
{results}

Write a clear, concise final answer. Include relevant data from the results. \
Do not list the steps — just give the answer.
"""

_REFLECTION_TEMPLATE = """\
Review whether your result fully completes the task.

TASK: {prompt}
YOUR RESULT: {result}

If the result is complete and correct respond with exactly:
VERIFIED: <your result unchanged>

If you spot a problem or gap respond with:
CORRECTION: <improved result>

Be concise. Do not restate the task.
"""


# ── TF-IDF tool selection  (paper: arxiv 2509.20415) ─────────────────────────


def _select_tools(prompt: str, top_k: int | None = None) -> str:
    """
    Return the descriptions of the top-k most task-relevant tools.

    Scores each tool by the TF-IDF cosine similarity between the task
    prompt and the tool's description.  Falls back to all tools when
    there are fewer registered tools than top_k.
    """
    if top_k is None:
        top_k = settings.tool_top_k

    registry_items = list(_REGISTRY.items())
    if len(registry_items) <= top_k:
        return tools_description()

    def _tokenise(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    # Build TF vectors
    prompt_tokens = _tokenise(prompt)
    prompt_tf = Counter(prompt_tokens)

    # IDF: log((N+1)/(df+1)) over the tool corpus
    N = len(registry_items)
    corpus_tokens: list[list[str]] = [
        _tokenise(info["description"]) for _, info in registry_items
    ]
    df: Counter = Counter()
    for doc in corpus_tokens:
        df.update(set(doc))

    def _tfidf_vec(tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        return {
            t: (tf[t] / max(len(tokens), 1))
               * math.log((N + 1) / (df.get(t, 0) + 1))
            for t in tf
        }

    prompt_vec = _tfidf_vec(prompt_tokens)

    scores: list[tuple[float, str, str]] = []
    for (name, info), doc_tokens in zip(registry_items, corpus_tokens):
        doc_vec = _tfidf_vec(doc_tokens)
        # Cosine similarity
        common = set(prompt_vec) & set(doc_vec)
        dot = sum(prompt_vec[t] * doc_vec[t] for t in common)
        mag_p = math.sqrt(sum(v * v for v in prompt_vec.values())) or 1
        mag_d = math.sqrt(sum(v * v for v in doc_vec.values())) or 1
        scores.append((dot / (mag_p * mag_d), name, info["description"]))

    scores.sort(reverse=True)
    return "\n".join(
        f"  {name}: {desc}" for _, name, desc in scores[:top_k]
    )


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
        return r.json()["choices"][0]["message"]["content"].strip()


# ── Plan parser ───────────────────────────────────────────────────────────────


def _parse_plan(text: str) -> tuple[list[dict], str, str | None]:
    """
    Extract (tool_calls, final_answer, parse_error) from LLM output.

    Returns parse_error=None on success, or an error string if JSON
    could not be decoded (used by the plan-repair path).
    """
    final_answer = ""

    if "RESULT:" in text:
        before, _, after = text.partition("RESULT:")
        text         = before
        final_answer = after.strip()

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return [], final_answer, "No JSON array found in response"

    try:
        calls = json.loads(match.group())
        if isinstance(calls, list):
            return calls, final_answer, None
        return [], final_answer, f"Expected JSON array, got {type(calls).__name__}"
    except json.JSONDecodeError as e:
        return [], final_answer, str(e)


# ── Self-reflection helper (arxiv 2512.20845) ─────────────────────────────────


async def _reflect(prompt: str, result: str, log: Callable[[str], None]) -> str:
    """
    Ask the LLM to verify its own answer.
    Returns the original result (VERIFIED) or a corrected one (CORRECTION).
    """
    log("Self-reflecting on result...")
    reflection_prompt = _REFLECTION_TEMPLATE.format(prompt=prompt, result=result)
    try:
        reflection = await _call_ollama(reflection_prompt)
        if reflection.startswith("CORRECTION:"):
            corrected = reflection[len("CORRECTION:"):].strip()
            log("Self-reflection found an issue — applying correction")
            return corrected
        log("Self-reflection: result verified")
        # Strip the VERIFIED: prefix if present
        if reflection.startswith("VERIFIED:"):
            return reflection[len("VERIFIED:"):].strip()
    except Exception as ref_err:
        log(f"Self-reflection skipped: {ref_err}")
    return result


# ── Main agent loop ───────────────────────────────────────────────────────────


async def run_agent(
    prompt: str,
    log: Callable[[str], None],
    memory: str = "",
) -> str:
    """
    Plan and execute a task.

    Args:
        prompt:  natural-language task description
        log:     callback called after each significant step
        memory:  optional block of similar past results (task-memory feature);
                 injected into the planner prompt for context

    Returns:
        Final result string (always non-empty).
    """
    log(f"Task received: {prompt}")

    # ── Step 1: select relevant tools (TF-IDF) ────────────────────────────────
    selected_tools = _select_tools(prompt)
    memory_block   = f"SIMILAR PAST TASKS:\n{memory}\n\n" if memory else ""

    planner_prompt = _PLANNER_TEMPLATE.format(
        tools        = selected_tools,
        memory_block = memory_block,
        prompt       = prompt,
        max_calls    = settings.max_tool_calls,
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

    # ── Step 2: parse — repair once if malformed (arxiv 2505.24726) ──────────
    tool_calls, final_answer, parse_error = _parse_plan(plan_text)

    # Trigger repair when there's any parse error AND the LLM didn't give a
    # clean direct answer (i.e. the response looks like a failed JSON attempt).
    _looks_like_json_attempt = "[" in plan_text
    if parse_error and _looks_like_json_attempt:
        log(f"Plan JSON malformed ({parse_error}) — attempting repair...")
        repair_prompt = _REPAIR_TEMPLATE.format(
            error        = parse_error,
            bad_response = plan_text[:500],
        )
        try:
            repaired = await _call_ollama(repair_prompt)
            tool_calls, final_answer, parse_error = _parse_plan(repaired)
            if parse_error:
                log("Repair failed — falling back to direct answer")
                tool_calls = []
            else:
                log("Plan repaired successfully")
        except Exception as repair_err:
            log(f"Repair call failed: {repair_err}")
            tool_calls = []

    # ── Step 3: execute tool calls ────────────────────────────────────────────
    if not tool_calls:
        log("No tool calls — returning direct LLM answer")
        raw_result = final_answer or plan_text
        return await _reflect(prompt, raw_result, log)

    log(f"Executing {len(tool_calls)} tool call(s)...")
    tool_results: list[dict] = []

    for i, call in enumerate(tool_calls[: settings.max_tool_calls]):
        tool_name = call.get("tool", "")
        tool_args = call.get("args", {})
        reason    = call.get("reason", "")
        thought   = call.get("thought", "")

        args_preview = ", ".join(
            f"{k}={repr(v)[:50]}" for k, v in tool_args.items()
        )
        if thought:
            log(f"    Thought: {thought}")
        log(f"[{i + 1}/{len(tool_calls)}] {tool_name}({args_preview}) — {reason}")

        result = await call_tool(tool_name, tool_args)

        preview = result[:150] + ("..." if len(result) > 150 else "")
        log(f"    Result: {preview}")

        tool_results.append({"tool": tool_name, "args": tool_args, "result": result})

    # ── Step 4: synthesise ────────────────────────────────────────────────────
    if final_answer:
        log("Using LLM-provided final answer")
        raw_result = final_answer
    else:
        log("Synthesising final answer from tool results...")
        results_block = "\n".join(
            f"[{r['tool']}]: {r['result']}" for r in tool_results
        )
        synthesis_prompt = _SYNTHESIS_TEMPLATE.format(
            prompt  = prompt,
            results = results_block,
        )
        raw_result = await _call_ollama(synthesis_prompt)

    # ── Step 5: self-reflection (arxiv 2512.20845) ────────────────────────────
    raw_result = await _reflect(prompt, raw_result, log)
    log("Done.")
    return raw_result
