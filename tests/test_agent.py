"""
tests/test_agent.py — Unit tests for the agent planning + execution logic.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.agent import _parse_plan, run_agent


# ── _parse_plan ───────────────────────────────────────────────────────────────


def test_parse_plan_valid_json_array():
    text = '[{"tool": "http_get", "args": {"url": "https://x.com"}, "reason": "fetch"}]\nRESULT: done'
    calls, answer = _parse_plan(text)
    assert len(calls) == 1
    assert calls[0]["tool"] == "http_get"
    assert answer == "done"


def test_parse_plan_no_tool_calls():
    text = "[] \nRESULT: Nothing to do."
    calls, answer = _parse_plan(text)
    assert calls == []
    assert answer == "Nothing to do."


def test_parse_plan_missing_result_section():
    text = '[{"tool": "echo", "args": {}, "reason": "test"}]'
    calls, answer = _parse_plan(text)
    assert len(calls) == 1
    assert answer == ""


def test_parse_plan_malformed_json_returns_empty():
    calls, answer = _parse_plan("not json at all RESULT: fallback")
    assert calls == []
    assert answer == "fallback"


def test_parse_plan_extra_text_around_json():
    text = "Sure! Here is my plan:\n[{\"tool\": \"read_file\", \"args\": {\"filename\": \"x\"}, \"reason\": \"r\"}]\nRESULT:"
    calls, answer = _parse_plan(text)
    assert len(calls) == 1
    assert calls[0]["tool"] == "read_file"


# ── run_agent — happy path ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_agent_no_tool_calls():
    """LLM returns a direct answer with no tool calls — agent returns it as-is."""
    logs = []

    with patch("app.agent._call_ollama", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "[]\nRESULT: The answer is 42."
        result = await run_agent("What is the answer?", logs.append)

    assert result == "The answer is 42."
    assert any("direct" in l.lower() or "no tool" in l.lower() for l in logs)


@pytest.mark.asyncio
async def test_run_agent_single_tool_call():
    """LLM plans one tool call; agent executes it and synthesises result."""
    logs = []

    plan = '[{"tool": "echo_test", "args": {"text": "hello"}, "reason": "test"}]'

    with patch("app.agent._call_ollama", new_callable=AsyncMock) as mock_llm, \
         patch("app.agent.call_tool", new_callable=AsyncMock) as mock_tool:
        mock_llm.side_effect = [plan, "Synthesis: tool returned hello"]
        mock_tool.return_value = "hello"

        result = await run_agent("Echo hello", logs.append)

    assert "hello" in result.lower() or "synthesis" in result.lower()
    mock_tool.assert_called_once_with("echo_test", {"text": "hello"})


@pytest.mark.asyncio
async def test_run_agent_respects_max_tool_calls():
    """Agent never executes more than max_tool_calls steps."""
    logs = []

    # Give it 20 tool calls — max is 10
    calls = [{"tool": "read_file", "args": {"filename": "x"}, "reason": "r"}] * 20
    plan  = f"{calls}\nRESULT: done"

    import json
    plan = json.dumps(calls) + "\nRESULT: done"

    with patch("app.agent._call_ollama", new_callable=AsyncMock) as mock_llm, \
         patch("app.agent.call_tool", new_callable=AsyncMock) as mock_tool:
        mock_llm.return_value = plan
        mock_tool.return_value = "ok"

        await run_agent("Read a file 20 times", logs.append)

    from app.config import settings
    assert mock_tool.call_count <= settings.max_tool_calls


# ── run_agent — error paths ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_agent_ollama_unreachable():
    """run_agent raises RuntimeError with a helpful message when Ollama is down."""
    import httpx

    logs = []

    with patch("app.agent._call_ollama", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
            await run_agent("Do something", logs.append)


@pytest.mark.asyncio
async def test_run_agent_logs_each_step():
    """Every tool execution step is logged."""
    logs = []

    plan = '[{"tool": "write_file", "args": {"filename": "a.txt", "content": "hi"}, "reason": "save"}]'

    with patch("app.agent._call_ollama", new_callable=AsyncMock) as mock_llm, \
         patch("app.agent.call_tool", new_callable=AsyncMock) as mock_tool:
        mock_llm.side_effect = [plan, "Saved."]
        mock_tool.return_value = "Wrote 2 bytes"

        await run_agent("Save hi to a file", logs.append)

    # At least the task receipt, planning, and tool execution should be logged
    assert len(logs) >= 3
    assert any("write_file" in l for l in logs)
