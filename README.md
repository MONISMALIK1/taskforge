# TaskForge

An AI agent backend that accepts plain-English tasks and executes them automatically using real tools — HTTP requests, file I/O, shell commands, and more.

POST a task, get back a result. No prompt engineering required.

---

## How it works

```
POST /tasks  {"prompt": "Fetch the weather API and save the result to weather.txt"}
             ↓
         LLM (Ollama) plans the tool calls
             ↓
         Agent executes: http_get → write_file
             ↓
         LLM synthesises a final answer
             ↓
GET /tasks/{id}  → { status: "done", result: "Saved 412 bytes to weather.txt" }
```

The agent loop:
1. Sends the task to a local Ollama model with a list of available tools
2. The LLM returns a JSON array of tool calls + an optional `RESULT:` section
3. The agent executes each tool call in order, collecting results
4. If no `RESULT:` was provided, it sends tool outputs back to the LLM for synthesis
5. All steps are logged incrementally to the database in real time

---

## Quick start

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) running locally with a model pulled:

```bash
ollama pull llama3.2
ollama serve          # runs on http://localhost:11434
```

### Install and run

```bash
git clone https://github.com/MONISMALIK1/taskforge.git
cd taskforge

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env        # edit if needed
uvicorn app.main:app --reload
```

The API is now live at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## API reference

### `POST /tasks`
Submit a new task. Returns `202 Accepted` immediately.

```json
{ "prompt": "Fetch https://httpbin.org/get and summarise the response" }
```

Response:
```json
{
  "id": "3f2e1d...",
  "prompt": "...",
  "status": "pending",
  "result": null,
  "logs": [],
  "created_at": "2026-05-21T10:00:00Z",
  "updated_at": "2026-05-21T10:00:00Z"
}
```

### `GET /tasks/{id}`
Poll for status and results.

`status` is one of: `pending` | `running` | `done` | `failed`

### `GET /tasks?limit=20&offset=0`
List all tasks, newest first. `limit` max is 100.

### `DELETE /tasks/{id}`
Remove a task record. Returns `204 No Content`.

### `GET /health`
Liveness check — returns Ollama endpoint and model name.

---

## Available tools

| Tool | Description |
|------|-------------|
| `http_get` | Fetch a URL. Returns up to 4 KB of response body. |
| `write_file` | Write text to `/tmp/<filename>`. |
| `read_file` | Read text from `/tmp/<filename>`. Returns up to 4 KB. |
| `run_shell` | Run a whitelisted shell command in `/tmp`. |
| `parse_json` | Pretty-print a raw JSON string. |
| `summarise_text` | Return the first 1 KB of a long text string. |

Shell command whitelist: `cat`, `curl`, `cut`, `date`, `echo`, `find`, `grep`, `head`, `ls`, `pip`, `python3`, `pwd`, `sort`, `tail`, `uniq`, `wc`

---

## Configuration

All settings are read from environment variables (or a `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_ENDPOINT` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_MODEL` | `llama3.2` | Model name |
| `OLLAMA_TIMEOUT` | `60` | Per-request timeout (seconds) |
| `MAX_TOOL_CALLS` | `10` | Max tool calls per agent run |
| `TASK_TIMEOUT` | `300` | Hard task timeout (seconds) |
| `DATABASE_URL` | `sqlite:///./taskforge.db` | SQLAlchemy database URL |

---

## Docker

```bash
# Build
docker build -t taskforge .

# Run (Ollama on the host)
docker run -p 8000:8000 \
  -e OLLAMA_ENDPOINT=http://host.docker.internal:11434 \
  -v taskforge-data:/data \
  taskforge
```

---

## Running tests

```bash
pytest -v
```

Tests use an in-memory SQLite database and mock Ollama/HTTP calls — no live services needed.

---

## Project structure

```
taskforge/
├── app/
│   ├── agent.py      # LLM planning + agent execution loop
│   ├── config.py     # pydantic-settings configuration
│   ├── database.py   # SQLAlchemy engine + session factory
│   ├── main.py       # FastAPI routes + background worker
│   ├── models.py     # ORM models + Pydantic schemas
│   └── tools.py      # Tool registry + all tool implementations
├── tests/
│   ├── test_agent.py # Agent loop unit tests
│   ├── test_api.py   # FastAPI integration tests
│   └── test_tools.py # Tool unit tests
├── .env.example
├── Dockerfile
├── requirements.txt
└── README.md
```
