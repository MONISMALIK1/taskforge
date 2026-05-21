"""
tests/test_tools.py — Unit tests for all agent tools.
"""

import json
from pathlib import Path

import pytest
import respx
import httpx

from app.tools import call_tool, tools_description


# ── http_get ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_http_get_success():
    respx.get("https://example.com").mock(
        return_value=httpx.Response(200, text="Hello World")
    )
    result = await call_tool("http_get", {"url": "https://example.com"})
    assert "Hello World" in result


@pytest.mark.asyncio
@respx.mock
async def test_http_get_non_200():
    respx.get("https://example.com/missing").mock(
        return_value=httpx.Response(404)
    )
    result = await call_tool("http_get", {"url": "https://example.com/missing"})
    assert "404" in result


@pytest.mark.asyncio
@respx.mock
async def test_http_get_truncates_long_response():
    big_body = "x" * 10_000
    respx.get("https://example.com/big").mock(
        return_value=httpx.Response(200, text=big_body)
    )
    result = await call_tool("http_get", {"url": "https://example.com/big"})
    assert len(result) <= 4096


# ── write_file / read_file ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_and_read_file():
    result = await call_tool("write_file", {"filename": "tf_test.txt", "content": "hello taskforge"})
    assert "tf_test.txt" in result

    content = await call_tool("read_file", {"filename": "tf_test.txt"})
    assert content == "hello taskforge"


@pytest.mark.asyncio
async def test_write_file_strips_path_traversal():
    # ../../etc/passwd should be sandboxed to /tmp/passwd
    await call_tool("write_file", {"filename": "../../etc/tf_safe.txt", "content": "safe"})
    assert not Path("/etc/tf_safe.txt").exists()
    assert Path("/tmp/tf_safe.txt").exists()
    Path("/tmp/tf_safe.txt").unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_read_file_not_found():
    result = await call_tool("read_file", {"filename": "nonexistent_xyz.txt"})
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_read_file_truncates_large_file(tmp_path):
    big = "y" * 10_000
    (Path("/tmp") / "tf_big.txt").write_text(big)
    result = await call_tool("read_file", {"filename": "tf_big.txt"})
    assert len(result) <= 4096
    Path("/tmp/tf_big.txt").unlink(missing_ok=True)


# ── run_shell ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_shell_echo():
    result = await call_tool("run_shell", {"command": "echo hello"})
    assert "hello" in result


@pytest.mark.asyncio
async def test_run_shell_blocks_disallowed_command():
    result = await call_tool("run_shell", {"command": "rm -rf /tmp"})
    assert "not in the allowed" in result.lower() or "not allowed" in result.lower()


@pytest.mark.asyncio
async def test_run_shell_empty_command():
    result = await call_tool("run_shell", {"command": ""})
    assert "empty" in result.lower() or "command" in result.lower()


# ── parse_json ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_json_valid():
    result = await call_tool("parse_json", {"text": '{"key": "value", "num": 42}'})
    assert '"key"' in result
    assert '"value"' in result


@pytest.mark.asyncio
async def test_parse_json_invalid():
    result = await call_tool("parse_json", {"text": "not json at all"})
    assert "invalid" in result.lower() or "json" in result.lower()


# ── summarise_text ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_summarise_text_short():
    result = await call_tool("summarise_text", {"text": "short"})
    assert result == "short"


@pytest.mark.asyncio
async def test_summarise_text_long():
    long_text = "a" * 2000
    result = await call_tool("summarise_text", {"text": long_text})
    assert "truncated" in result
    assert len(result) < 2000


# ── Unknown tool ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    result = await call_tool("fly_to_moon", {})
    assert "unknown" in result.lower() or "fly_to_moon" in result


# ── tools_description ─────────────────────────────────────────────────────────


def test_tools_description_lists_all_tools():
    desc = tools_description()
    for name in ["http_get", "write_file", "read_file", "run_shell", "parse_json", "summarise_text", "list_files"]:
        assert name in desc


@pytest.mark.asyncio
async def test_list_files_returns_filenames(tmp_path):
    (Path("/tmp") / "tf_listtest.txt").write_text("hello")
    result = await call_tool("list_files", {"pattern": "tf_listtest.txt"})
    assert "tf_listtest.txt" in result
    Path("/tmp/tf_listtest.txt").unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_list_files_no_match():
    result = await call_tool("list_files", {"pattern": "no_such_file_xyz_*.txt"})
    assert "no files" in result.lower() or "no_such_file" in result.lower()
