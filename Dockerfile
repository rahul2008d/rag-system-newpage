FROM python:3.13-slim AS builder
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_CACHE_DIR=/tmp/uv-cache

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project && \
    rm -rf /tmp/uv-cache /root/.cache

COPY src ./src

FROM python:3.13-slim AS runtime
WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STORE_DIR=vector_store

RUN mkdir -p /app/vector_store /app/documents

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
