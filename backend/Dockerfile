FROM python:3.11-slim

WORKDIR /app
# PIP_NO_CACHE_DIR is intentionally not set: caching goes to a BuildKit cache
# mount instead, so warm rebuilds reuse downloads without bloating the layer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime-only requirements (no pytest/mypy/ruff/httpx); kept as the first
# layer so backend code changes don't invalidate the heavy pip install.
COPY backend/requirements.txt backend/requirements-runtime.txt ./backend/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r backend/requirements-runtime.txt

COPY backend ./backend
RUN pip install --no-deps ./backend

EXPOSE 8000
CMD ["uvicorn", "friday.main:app", "--host", "0.0.0.0", "--port", "8000"]
