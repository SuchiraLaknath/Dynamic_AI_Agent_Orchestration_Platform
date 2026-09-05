FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY backend /srv/backend
# MCP servers run as stdio subprocesses of the backend, so their source has to
# be inside this image rather than a container of their own.
COPY mcp_servers /srv/mcp_servers

RUN pip install --upgrade pip && pip install -e "/srv/backend[dev]"

WORKDIR /srv/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
