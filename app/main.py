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
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
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

        # Run with a hard timeout
        result = await asyncio.wait_for(
            run_agent(prompt, _log),
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
    db.delete(record)
    db.commit()


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
