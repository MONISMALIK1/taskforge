"""
tools.py — All tools available to the TaskForge agent.

Each tool is a plain async function registered via @tool().
Tools are intentionally sandboxed:
  - file ops are limited to /tmp
  - shell commands are whitelist-only
  - HTTP responses are capped at 4 KB to avoid flooding the context window
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

import httpx

# Shell metacharacters that enable command injection.
# Semicolons chain commands, pipes redirect output to arbitrary binaries,
# $() / backticks allow subshell execution, & forks background processes.
_SHELL_INJECTION_RE = re.compile(r"[;&|`${}]")


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, dict[str, Any]] = {}


def tool(name: str, description: str) -> Callable:
    """Decorator that registers a function as an agent tool."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = {"fn": fn, "description": description}
        return fn
    return decorator


# ── Tool implementations ──────────────────────────────────────────────────────


@tool(
    "http_get",
    "Fetch the text content of a URL. "
    "Args: url (str). Returns up to 4 KB of response body.",
)
async def http_get(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "TaskForge-Agent/1.0"})
            r.raise_for_status()
            return r.text[:4096]
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code}: {e.request.url}"
    except httpx.RequestError as e:
        return f"Request failed: {e}"


@tool(
    "write_file",
    "Write text to a file inside /tmp. "
    "Args: filename (str), content (str). Returns a confirmation.",
)
async def write_file(filename: str, content: str) -> str:
    # Strip any path traversal — only the basename is used
    safe_name = Path(filename).name
    path = Path("/tmp") / safe_name
    # Run blocking I/O in a thread so the event loop stays free
    await asyncio.to_thread(path.write_text, content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {path}"


@tool(
    "read_file",
    "Read text from a file inside /tmp. "
    "Args: filename (str). Returns up to 4 KB of content.",
)
async def read_file(filename: str) -> str:
    safe_name = Path(filename).name
    path = Path("/tmp") / safe_name
    if not await asyncio.to_thread(path.exists):
        return f"File not found: {safe_name}"
    content = await asyncio.to_thread(path.read_text, encoding="utf-8")
    return content[:4096]


# Shell commands that are safe to run
_ALLOWED_COMMANDS = {
    "cat", "echo", "ls", "wc", "grep", "sort",
    "head", "tail", "curl", "python3", "pip",
    "date", "pwd", "find", "uniq", "cut",
}


@tool(
    "run_shell",
    "Run a whitelisted shell command inside /tmp. "
    f"Allowed commands: {', '.join(sorted(_ALLOWED_COMMANDS))}. "
    "Args: command (str). Returns stdout + stderr (up to 2 KB).",
)
async def run_shell(command: str) -> str:
    parts = command.strip().split()
    if not parts:
        return "Empty command."
    binary = parts[0]
    if binary not in _ALLOWED_COMMANDS:
        return (
            f"'{binary}' is not in the allowed command list. "
            f"Allowed: {', '.join(sorted(_ALLOWED_COMMANDS))}"
        )
    # Block shell injection: semicolons, pipes, subshell operators etc.
    # These bypass the binary whitelist by chaining arbitrary commands.
    if _SHELL_INJECTION_RE.search(command):
        return (
            "Command contains disallowed shell metacharacters (; & | ` $ { }). "
            "Only simple commands are permitted."
        )
    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
            cwd="/tmp",
        )
        output = (proc.stdout + proc.stderr).strip()
        return output[:2048] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 15 seconds."
    except Exception as e:
        return f"Shell error: {e}"


@tool(
    "parse_json",
    "Parse a raw JSON string and return it pretty-printed (up to 3 KB). "
    "Useful for inspecting API responses. Args: text (str).",
)
async def parse_json(text: str) -> str:
    try:
        data = json.loads(text)
        return json.dumps(data, indent=2, ensure_ascii=False)[:3072]
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"


@tool(
    "summarise_text",
    "Return the first 1 KB of a long text string — useful for extracting "
    "the beginning of a large document or API response. Args: text (str).",
)
async def summarise_text(text: str) -> str:
    truncated = text[:1024]
    suffix = f"... [{len(text) - 1024} chars truncated]" if len(text) > 1024 else ""
    return truncated + suffix


@tool(
    "list_files",
    "List files in /tmp, optionally filtered by a glob pattern. "
    "Args: pattern (str, default '*'). Returns a newline-separated list of filenames.",
)
async def list_files(pattern: str = "*") -> str:
    matches = sorted(p for p in Path("/tmp").glob(pattern) if p.is_file())
    if not matches:
        return f"No files matching '{pattern}' in /tmp."
    return "\n".join(f.name for f in matches)


# ── Public helpers ────────────────────────────────────────────────────────────


def tools_description() -> str:
    """Return a formatted list of all available tools for the LLM prompt."""
    return "\n".join(
        f"  {name}: {info['description']}"
        for name, info in _REGISTRY.items()
    )


async def call_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a named tool with the given args. Returns a string result."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY)
        return f"Unknown tool '{name}'. Available tools: {available}"
    try:
        return await _REGISTRY[name]["fn"](**args)
    except TypeError as e:
        return f"Bad arguments for '{name}': {e}"
    except Exception as e:
        return f"Tool '{name}' raised an error: {e}"
