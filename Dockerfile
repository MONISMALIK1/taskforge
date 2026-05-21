# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL org.opencontainers.image.title="TaskForge"
LABEL org.opencontainers.image.description="AI agent backend — POST a plain-English task, get results."

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source
COPY app/ ./app/

# SQLite database lives in /data so it can be volume-mounted
RUN mkdir -p /data
ENV DATABASE_URL="sqlite:////data/taskforge.db"

# Ollama endpoint — override at runtime with:
#   docker run -e OLLAMA_ENDPOINT=http://host.docker.internal:11434 ...
ENV OLLAMA_ENDPOINT="http://host.docker.internal:11434"
ENV OLLAMA_MODEL="llama3.2"

EXPOSE 8000

# Non-root user for security
RUN adduser --disabled-password --gecos "" taskforge
USER taskforge

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
