"""
main.py — FastAPI application entry point.

Routes:
  POST   /tasks          — submit a new task (returns 202 immediately)
  GET    /tasks          — list tasks (paginated)
  GET    /tasks/{id}     — get task status, logs, and result
  DELETE /tasks/{id}     — delete a task record
  GET    /health         — liveness check
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session

from .agent import run_agent
from .config import settings
from .database import Base, SessionLocal, engine, get_db
from .models import TaskCreate, TaskRecord, TaskResponse, TaskStatus
from .tools import _REGISTRY


# ── App lifecycle ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="TaskForge",
    description=(
        "AI agent backend — POST a plain-English task, "
        "the agent plans it and runs it automatically using real tools."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Background worker ─────────────────────────────────────────────────────────


def _build_memory(db, current_task_id: str, limit: int = 3) -> str:
    """
    Task memory (A-Mem, arxiv 2502.12110).

    Fetch the last `limit` successfully completed tasks (excluding the
    current one) and format them as a brief context block for the agent.
    Simple word-overlap scoring keeps this dependency-free.
    """
    past = (
        db.query(TaskRecord)
        .filter(
            TaskRecord.id     != current_task_id,
            TaskRecord.status == TaskStatus.done,
            TaskRecord.result.isnot(None),
        )
        .order_by(TaskRecord.updated_at.desc())
        .limit(limit)
        .all()
    )
    if not past:
        return ""
    lines = [
        f'  - Task: "{r.prompt[:80]}" → Result: "{(r.result or "")[:120]}"'
        for r in past
    ]
    return "\n".join(lines)


async def _execute_task(task_id: str, prompt: str) -> None:
    """
    Run the agent for a task in the background.
    Writes incremental logs to the DB after each agent step.
    Updates final status to done/failed when complete.
    """
    db   = SessionLocal()
    logs: list[str] = []

    def _log(message: str) -> None:
        logs.append(message)
        record = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if record:
            record.logs      = logs.copy()
            record.updated_at = datetime.now(timezone.utc)
            db.commit()

    try:
        # Mark as running
        record = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        record.status     = TaskStatus.running
        record.updated_at = datetime.now(timezone.utc)
        db.commit()

        # Build task memory from past completed tasks
        memory = _build_memory(db, task_id)
        if memory:
            _log(f"Memory: injecting {len(memory.splitlines())} past task(s) as context")

        # Run with a hard timeout
        result = await asyncio.wait_for(
            run_agent(prompt, _log, memory=memory),
            timeout=settings.task_timeout,
        )

        # Mark as done
        record            = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        record.status     = TaskStatus.done
        record.result     = result
        record.updated_at = datetime.now(timezone.utc)
        db.commit()

    except asyncio.TimeoutError:
        _log(f"Task timed out after {settings.task_timeout} seconds.")
        record            = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if record:
            record.status     = TaskStatus.failed
            record.result     = "Task timed out."
            record.updated_at = datetime.now(timezone.utc)
            db.commit()

    except Exception as exc:
        _log(f"Unhandled error: {exc}")
        record            = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if record:
            record.status     = TaskStatus.failed
            record.result     = str(exc)
            record.updated_at = datetime.now(timezone.utc)
            db.commit()

    finally:
        db.close()


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post(
    "/tasks",
    response_model=TaskResponse,
    status_code=202,
    summary="Submit a new task",
    description=(
        "Accepts a plain-English task description. "
        "Returns 202 immediately with the task ID. "
        "Poll `GET /tasks/{id}` to check progress."
    ),
)
async def create_task(
    body: TaskCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TaskRecord:
    # Concurrency cap — prevent flooding Ollama with simultaneous tasks
    active = (
        db.query(func.count(TaskRecord.id))
        .filter(TaskRecord.status.in_([TaskStatus.pending, TaskStatus.running]))
        .scalar()
    )
    if active >= settings.max_concurrent_tasks:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Too many active tasks ({active}/{settings.max_concurrent_tasks}). "
                "Wait for a task to finish before submitting another."
            ),
        )

    task_id = str(uuid.uuid4())
    record  = TaskRecord(
        id     = task_id,
        prompt = body.prompt,
        status = TaskStatus.pending,
        logs   = [],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    background_tasks.add_task(_execute_task, task_id, body.prompt)
    return record


@app.get(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    summary="Get task status and result",
)
def get_task(
    task_id: str,
    db: Session = Depends(get_db),
) -> TaskRecord:
    record = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return record


@app.get(
    "/tasks",
    response_model=list[TaskResponse],
    summary="List tasks (newest first)",
)
def list_tasks(
    limit:  int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0,  ge=0),
    db: Session = Depends(get_db),
) -> list[TaskRecord]:
    return (
        db.query(TaskRecord)
        .order_by(TaskRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


@app.delete(
    "/tasks/{task_id}",
    status_code=204,
    summary="Delete a task record",
)
def delete_task(
    task_id: str,
    db: Session = Depends(get_db),
) -> None:
    record = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    # Guard: deleting a running/pending task would orphan the background worker —
    # it would keep running with nowhere to write logs or final status.
    if record.status in (TaskStatus.running, TaskStatus.pending):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot delete task '{task_id}' while it is '{record.status}'. "
                "Wait for it to finish, or let it time out first."
            ),
        )
    db.delete(record)
    db.commit()


@app.post(
    "/tasks/{task_id}/retry",
    response_model=TaskResponse,
    status_code=202,
    summary="Retry a failed or completed task",
    description=(
        "Creates a new task record with the same prompt and queues it for execution. "
        "Only tasks with status `failed` or `done` can be retried. "
        "The original task record is left unchanged."
    ),
)
async def retry_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TaskRecord:
    original = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
    if not original:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    if original.status not in (TaskStatus.failed, TaskStatus.done):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Task is currently '{original.status}' — "
                "only failed or done tasks can be retried."
            ),
        )
    new_id = str(uuid.uuid4())
    record  = TaskRecord(
        id     = new_id,
        prompt = original.prompt,
        status = TaskStatus.pending,
        logs   = [],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    background_tasks.add_task(_execute_task, new_id, original.prompt)
    return record


@app.get(
    "/stats",
    summary="Task statistics",
    description="Returns total task count and a breakdown by status.",
)
def stats(db: Session = Depends(get_db)) -> dict:
    rows = (
        db.query(TaskRecord.status, func.count(TaskRecord.id))
        .group_by(TaskRecord.status)
        .all()
    )
    counts: dict = {s.value: 0 for s in TaskStatus}
    for status, count in rows:
        counts[status] = count
    counts["total"] = sum(counts.values())
    return counts


@app.get(
    "/tools",
    summary="List available agent tools",
    description="Returns every tool the agent can call, with its name and description.",
)
def list_tools() -> list[dict]:
    return [
        {"name": name, "description": info["description"]}
        for name, info in _REGISTRY.items()
    ]


@app.get(
    "/health",
    summary="Liveness check with Ollama connectivity",
    description=(
        "Returns service status plus whether Ollama is reachable. "
        "`ollama_reachable: false` means the agent will fail — "
        "run `ollama serve` to fix it."
    ),
)
async def health() -> dict:
    ollama_reachable = False
    try:
        async with httpx.AsyncClient(timeout=3) as _client:
            r = await _client.get(f"{settings.ollama_endpoint}/api/tags")
            ollama_reachable = r.status_code == 200
    except Exception:
        pass

    return {
        "status":           "ok",
        "model":            settings.ollama_model,
        "endpoint":         settings.ollama_endpoint,
        "ollama_reachable": ollama_reachable,
    }
