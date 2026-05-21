"""
tests/test_api.py — Integration tests for the FastAPI routes.

Uses FastAPI's TestClient (synchronous HTTPX wrapper) and an in-memory
SQLite database so tests are fully isolated and require no running server.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models import TaskStatus


# ── In-memory test database ───────────────────────────────────────────────────
# StaticPool keeps all connections on the same in-memory SQLite database;
# without it each new connection gets a blank DB and "no such table" errors.

TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=test_engine
)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_db():
    """
    Create all tables before each test, drop them after.

    We patch both:
    - `get_db` dependency (used by route handlers)
    - `app.main.SessionLocal` (used directly by the background worker _execute_task)
    so that every DB access goes through the in-memory test engine.
    """
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    with patch("app.main.SessionLocal", TestingSessionLocal):
        yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


# ── GET /health ───────────────────────────────────────────────────────────────


def test_health_returns_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model" in body
    assert "endpoint" in body


# ── GET /tools ────────────────────────────────────────────────────────────────


# ── GET /stats ────────────────────────────────────────────────────────────────


def test_stats_empty(client: TestClient):
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["pending"] == 0
    assert body["done"] == 0


def test_stats_counts_tasks(client: TestClient):
    with patch("app.main.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = "done"
        client.post("/tasks", json={"prompt": "Stats test task one here"})
        client.post("/tasks", json={"prompt": "Stats test task two here"})

    r = client.get("/stats")
    body = r.json()
    assert body["total"] == 2
    assert body["done"] == 2


# ── GET /tools ────────────────────────────────────────────────────────────────


def test_list_tools_returns_all_tools(client: TestClient):
    r = client.get("/tools")
    assert r.status_code == 200
    tools = r.json()
    names = {t["name"] for t in tools}
    expected = {"http_get", "write_file", "read_file", "run_shell",
                "parse_json", "summarise_text", "list_files"}
    assert expected.issubset(names)


def test_list_tools_each_has_description(client: TestClient):
    r = client.get("/tools")
    for tool in r.json():
        assert "name" in tool
        assert "description" in tool
        assert len(tool["description"]) > 10


# ── POST /tasks ───────────────────────────────────────────────────────────────


def test_create_task_returns_202(client: TestClient):
    r = client.post("/tasks", json={"prompt": "Fetch the homepage of example.com"})
    assert r.status_code == 202
    body = r.json()
    assert "id" in body
    assert body["status"] == TaskStatus.pending
    assert body["prompt"] == "Fetch the homepage of example.com"


def test_create_task_short_prompt_rejected(client: TestClient):
    r = client.post("/tasks", json={"prompt": "Hi"})
    assert r.status_code == 422


def test_create_task_missing_prompt_rejected(client: TestClient):
    r = client.post("/tasks", json={})
    assert r.status_code == 422


def test_create_task_assigns_unique_ids(client: TestClient):
    r1 = client.post("/tasks", json={"prompt": "First task to run"})
    r2 = client.post("/tasks", json={"prompt": "Second task to run"})
    assert r1.json()["id"] != r2.json()["id"]


# ── GET /tasks/{id} ───────────────────────────────────────────────────────────


def test_get_task_returns_record(client: TestClient):
    created = client.post("/tasks", json={"prompt": "Read some data from disk"}).json()
    r = client.get(f"/tasks/{created['id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == created["id"]
    assert body["prompt"] == "Read some data from disk"


def test_get_task_not_found(client: TestClient):
    r = client.get("/tasks/nonexistent-id-12345")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


# ── GET /tasks ────────────────────────────────────────────────────────────────


def test_list_tasks_empty(client: TestClient):
    r = client.get("/tasks")
    assert r.status_code == 200
    assert r.json() == []


def test_list_tasks_returns_all(client: TestClient):
    client.post("/tasks", json={"prompt": "First task for listing"})
    client.post("/tasks", json={"prompt": "Second task for listing"})
    r = client.get("/tasks")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_list_tasks_newest_first(client: TestClient):
    client.post("/tasks", json={"prompt": "Older task created first"})
    client.post("/tasks", json={"prompt": "Newer task created second"})
    r = client.get("/tasks")
    tasks = r.json()
    assert tasks[0]["prompt"] == "Newer task created second"


def test_list_tasks_pagination(client: TestClient):
    for i in range(5):
        client.post("/tasks", json={"prompt": f"Pagination test task number {i}"})
    r = client.get("/tasks?limit=2&offset=0")
    assert len(r.json()) == 2
    r2 = client.get("/tasks?limit=2&offset=2")
    assert len(r2.json()) == 2
    # IDs should differ
    ids1 = {t["id"] for t in r.json()}
    ids2 = {t["id"] for t in r2.json()}
    assert ids1.isdisjoint(ids2)


def test_list_tasks_invalid_limit(client: TestClient):
    r = client.get("/tasks?limit=0")
    assert r.status_code == 422


def test_list_tasks_limit_max(client: TestClient):
    r = client.get("/tasks?limit=101")
    assert r.status_code == 422


# ── DELETE /tasks/{id} ────────────────────────────────────────────────────────


def test_delete_task_returns_204(client: TestClient):
    created = client.post("/tasks", json={"prompt": "Task to be deleted soon"}).json()
    r = client.delete(f"/tasks/{created['id']}")
    assert r.status_code == 204


def test_delete_task_removes_record(client: TestClient):
    created = client.post("/tasks", json={"prompt": "Task that will be gone"}).json()
    client.delete(f"/tasks/{created['id']}")
    r = client.get(f"/tasks/{created['id']}")
    assert r.status_code == 404


def test_delete_task_not_found(client: TestClient):
    r = client.delete("/tasks/ghost-id-99999")
    assert r.status_code == 404


# ── Background task integration ───────────────────────────────────────────────


def test_task_transitions_to_done(client: TestClient):
    """
    Submit a task, mock run_agent to return immediately, confirm the
    record transitions from pending → done with a result.

    TestClient runs the event loop synchronously so background tasks
    finish before the fixture tears down.
    """
    with patch("app.main.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = "42 is the answer."
        created = client.post(
            "/tasks", json={"prompt": "What is the ultimate answer?"}
        ).json()
        task_id = created["id"]

    r = client.get(f"/tasks/{task_id}")
    body = r.json()
    assert body["status"] == TaskStatus.done
    assert body["result"] == "42 is the answer."


def test_task_transitions_to_failed_on_error(client: TestClient):
    """If run_agent raises, the task should end up in failed state."""
    with patch("app.main.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.side_effect = RuntimeError("Ollama is down")
        created = client.post(
            "/tasks", json={"prompt": "This task will unfortunately fail"}
        ).json()
        task_id = created["id"]

    r = client.get(f"/tasks/{task_id}")
    body = r.json()
    assert body["status"] == TaskStatus.failed
    assert "Ollama is down" in (body["result"] or "")


def test_task_logs_are_persisted(client: TestClient):
    """Logs written via the _log callback must appear in the GET response."""
    async def _fake_agent(prompt: str, log):
        log("Step 1: planning")
        log("Step 2: executing")
        return "Done."

    with patch("app.main.run_agent", side_effect=_fake_agent):
        created = client.post(
            "/tasks", json={"prompt": "Multi-step task with logging"}
        ).json()
        task_id = created["id"]

    r = client.get(f"/tasks/{task_id}")
    logs = r.json()["logs"]
    assert any("planning" in l for l in logs)
    assert any("executing" in l for l in logs)
