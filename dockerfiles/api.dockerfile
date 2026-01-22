# Base image with Python 3.13 on Debian Bookworm slim
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY uv.lock pyproject.toml ./
COPY configs/ configs/

RUN uv sync --frozen --no-install-project

COPY src/ src/
COPY README.md README.md
COPY LICENSE LICENSE
COPY reports/ reports/

COPY models/ models/

RUN uv sync --frozen

ENTRYPOINT ["/bin/sh", "-c", "uv run src/stuperml/data.py && uv run uvicorn src.stuperml.api:app --host 0.0.0.0 --port 8080"]