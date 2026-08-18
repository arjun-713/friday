FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY backend/requirements.txt backend/requirements-cpu.txt backend/requirements-dev.txt ./backend/
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r backend/requirements-cpu.txt

COPY backend ./backend
RUN pip install --no-deps ./backend

EXPOSE 8000
CMD ["uvicorn", "copilot.main:app", "--host", "0.0.0.0", "--port", "8000"]
