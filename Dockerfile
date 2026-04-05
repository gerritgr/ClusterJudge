FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade pip uv \
    && useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./

USER appuser

RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=appuser:appuser main.ipynb ./

RUN mkdir -p figures logs

EXPOSE 8888

CMD ["/app/.venv/bin/jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=", "--ServerApp.password="]
