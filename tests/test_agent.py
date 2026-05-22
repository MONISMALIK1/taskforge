"""
tests/test_agent.py — Unit tests for the agent planning + execution logic.

Covers all five research-paper features added to agent.py:
  1. Plan repair     (malformed JSON → repair-prompt retry)
  2. ReAct thoughts  (thought field logged)
  3. Self-reflection (VERIFIED / CORRECTION path)
  4. TF-IDF tool selection (_select_tools)
  5. Task memory     (memory kwarg injected into prompt)
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agent import _parse_plan, _select_tools, run_agent


# ── _parse_plan ───────────────────────────────────────────────────────────────


def test_parse_plan_valid_json_array():
    text = '[{"tool": "http_get", "args": {"url": "https://x.com"}, "reason": "fetch"}]\nRESULT: done'
    calls, answer, err = _parse_plan(text)
    assert len(calls) == 1
    assert calls[0]["tool"] == "http_get"
    assert answer == "done"
    assert err is None


def test_parse_plan_no_tool_calls():
    text = "[] \nRESULT: Nothing to do."
    calls, answer, err = _parse_plan(text)
    assert calls == []
    assert answer == "Nothing to do."
    assert err is None


def test_parse_plan_missing_result_section():
    text = '[{"tool": "echo", "args": {}, "reason": "test"}]'
    calls, answer, err = _parse_plan(text)
    assert len(calls) == 1
    assert answer == ""
    assert err is None


def test_parse_plan_malformed_json_returns_error():
    calls, answer, err = _parse_plan("[not valid json RESULT: fallback")
    assert calls == []
    assert answer == "fallback"
    assert err is not None   # parse error reported


def test_parse_plan_no_array_returns_error():
    calls, answer, err = _parse_plan("no array here RESULT: fallback")
    assert calls == []
    assert err is not None


def test_parse_plan_extra_text_around_json():
    text = 'Sure!\n[{"tool": "read_file", "args": {"filename": "x"}, "reason": "r"}]\nRESULT:'
    calls, answer, err = _parse_plan(text)
    assert len(calls) == 1
    assert calls[0]["tool"] == "read_file"
    assert err is None


# ── _select_tools (TF-IDF, paper: arxiv 2509.20415) ──────────────────────────


def test_select_tools_returns_string():
    result = _select_tools("fetch a URL and save the result")
    assert isinstance(result, str)
    assert len(result) > 0


def test_select_tools_http_task_includes_http_get():
    result = _select_tools("fetch the content of https://example.com")
    assert "http_get" in result


def test_select_tools_file_task_includes_file_tools():
    result = _select_tools("read and write a file to disk")
    assert "read_file" in result or "write_file" in result


def test_select_tools_respects_top_k():
    result = _select_tools("do something", top_k=2)
    # Each tool appears on its own line
    tool_lines = [l for l in result.splitlines() if l.strip()]
    assert len(tool_lines) <= 2


# ── run_agent — happy path ────────────────────────────────────────────────────


async def test_run_agent_no_tool_calls():
    """LLM returns a direct answer with no tool calls — agent returns it as-is."""
    logs = []

    # Planner returns empty array, reflection returns VERIFIED
    with patch("app.agent._call_ollama", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [
            "[]\nRESULT: The answer is 42.",   # plan
        ]
        result = await run_agent("What is the answer?", logs.append)

    assert result == "The answer is 42."
    assert any("direct" in l.lower() or "no tool" in l.lower() for l in logs)


async def test_run_agent_single_tool_call():
    """LLM plans one tool call; agent executes it, synthesises, then reflects."""
    logs = []
    plan = '[{"thought": "need to echo", "tool": "echo_test", "args": {"text": "hello"}, "reason": "test"}]'

    with patch("app.agent._call_ollama", new_callable=AsyncMock) as mock_llm, \
         patch("app.agent.call_tool", new_callable=AsyncMock) as mock_tool:
        mock_llm.side_effect = [
            plan,                              # planning
            "Synthesis: tool returned hello",  # synthesis
            "VERIFIED: tool returned hello",   # reflection
        ]
        mock_tool.return_value = "hello"
        result = await run_agent("Echo hello", logs.append)

    assert "hello" in result.lower() or "synthesis" in result.lower()
    mock_tool.assert_called_once_with("echo_test", {"text": "hello"})


async def test_run_agent_thought_field_is_logged():
    """The thought field from ReAct should appear in logs."""
    logs = []
    plan = '[{"thought": "I should fetch first", "tool": "http_get", "args": {"url": "https://x.com"}, "reason": "fetch"}]'

    with patch("app.agent._call_ollama", new_callable=AsyncMock) as mock_llm, \
         patch("app.agent.call_tool", new_callable=AsyncMock) as mock_tool:
        mock_llm.side_effect = [plan, "Result.", "VERIFIED: Result."]
        mock_tool.return_value = "page content"
        await run_agent("Fetch something", logs.append)

    assert any("I should fetch first" in l for l in logs)


async def test_run_agent_respects_max_tool_calls():
    """Agent never executes more than max_tool_calls steps."""
    logs = []
    calls = [{"tool": "read_file", "args": {"filename": "x"}, "reason": "r"}] * 20
    plan  = json.dumps(calls) + "\nRESULT: done"

    with patch("app.agent._call_ollama", new_callable=AsyncMock) as mock_llm, \
         patch("app.agent.call_tool", new_callable=AsyncMock) as mock_tool:
        mock_llm.return_value = plan
        mock_tool.return_value = "ok"
        await run_agent("Read a file 20 times", logs.append)

    from app.config import settings
    assert mock_tool.call_count <= settings.max_tool_calls


async def test_run_agent_memory_injected_into_prompt():
    """When memory is passed it should appear somewhere in the LLM call."""
    logs = []
    captured_prompts: list[str] = []

    async def fake_ollama(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "[]\nRESULT: done"

    with patch("app.agent._call_ollama", side_effect=fake_ollama):
        await run_agent(
            "Do something",
            logs.append,
            memory='  - Task: "old task" → Result: "old result"',
        )

    assert any("old task" in p for p in captured_prompts)


# ── run_agent — plan repair (arxiv 2505.24726) ────────────────────────────────


async def test_run_agent_repairs_malformed_plan():
    """If plan JSON is broken, agent re-prompts and uses the repaired plan."""
    logs = []

    bad_plan   = "[{broken json RESULT: oops"
    good_plan  = '[{"tool": "echo_test", "args": {"text": "repaired"}, "reason": "ok"}]'

    with patch("app.agent._call_ollama", new_callable=AsyncMock) as mock_llm, \
         patch("app.agent.call_tool", new_callable=AsyncMock) as mock_tool:
        mock_llm.side_effect = [
            bad_plan,                   # initial (malformed)
            good_plan,                  # repair response
            "All good.",                # synthesis
            "VERIFIED: All good.",      # reflection
        ]
        mock_tool.return_value = "ok"
        result = await run_agent("Do something", logs.append)

    assert any("repair" in l.lower() for l in logs)
    mock_tool.assert_called_once()


# ── run_agent — self-reflection (arxiv 2512.20845) ────────────────────────────


async def test_run_agent_applies_correction():
    """When reflection returns CORRECTION: the agent uses the improved answer."""
    logs = []

    with patch("app.agent._call_ollama", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [
            "[]\nRESULT: Incomplete answer.",    # plan
            "CORRECTION: Much better answer.",   # reflection
        ]
        result = await run_agent("Explain something", logs.append)

    assert result == "Much better answer."
    assert any("correction" in l.lower() for l in logs)


async def test_run_agent_keeps_result_when_verified():
    """When reflection returns VERIFIED: the original result is unchanged."""
    logs = []

    with patch("app.agent._call_ollama", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [
            "[]\nRESULT: Perfect answer.",
            "VERIFIED: Perfect answer.",
        ]
        result = await run_agent("Tell me something", logs.append)

    assert result == "Perfect answer."
    assert any("verified" in l.lower() for l in logs)


# ── run_agent — error paths ───────────────────────────────────────────────────


async def test_run_agent_ollama_unreachable():
    """run_agent raises RuntimeError with a helpful message when Ollama is down."""
    import httpx
    logs = []

    with patch("app.agent._call_ollama", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
            await run_agent("Do something", logs.append)


async def test_run_agent_logs_each_step():
    """Every tool execution step is logged."""
    logs = []
    plan = '[{"thought": "need to save", "tool": "write_file", "args": {"filename": "a.txt", "content": "hi"}, "reason": "save"}]'

    with patch("app.agent._call_ollama", new_callable=AsyncMock) as mock_llm, \
         patch("app.agent.call_tool", new_callable=AsyncMock) as mock_tool:
        mock_llm.side_effect = [plan, "Saved.", "VERIFIED: Saved."]
        mock_tool.return_value = "Wrote 2 bytes"
        await run_agent("Save hi to a file", logs.append)

    assert len(logs) >= 3
    assert any("write_file" in l for l in logs)
