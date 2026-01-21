# Base image with Python 3.13 on Debian Bookworm slim
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# install dependencies
COPY uv.lock pyproject.toml ./
COPY configs/ configs/

RUN uv sync --frozen --no-install-project

# copy source code
COPY src/ src/
COPY README.md README.md
COPY LICENSE LICENSE
COPY reports/ reports/

# copy artifacts (models, data) needed for the API
COPY models/ models/

# install the project itself
RUN uv sync --frozen

# define the command to run the API
# Cloud Run injects a $PORT environment variable, 8080 is the default.
# --host 0.0.0.0 makes it accessible outside the container.
ENTRYPOINT ["/bin/sh", "-c", "uv run src/stuperml/data.py && uv run uvicorn src.stuperml.api:app --host 0.0.0.0 --port 8080"]