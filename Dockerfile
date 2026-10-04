# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY evals ./evals
COPY config.example.yaml ./config.example.yaml

RUN pip install --no-cache-dir .

# Non-root user; /data holds the SQLite audit DB.
RUN useradd --create-home --uid 1000 triage \
    && mkdir -p /data \
    && chown -R triage:triage /data /app
USER triage

ENV TRIAGE_DB_PATH=/data/triage_agent.db \
    TRIAGE_CONFIG=/app/config.yaml \
    DRY_RUN=true

# Default: run the MCP server on stdio. Override for the CLI, e.g.:
#   docker run --env-file .env github-triage-agent \
#       triage-agent run --repo owner/project --issue 142 --dry-run
ENTRYPOINT ["triage-agent"]
CMD ["serve-mcp"]
