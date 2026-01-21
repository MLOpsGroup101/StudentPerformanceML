# Base image with Python 3.13 on Debian Bookworm slim
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY uv.lock pyproject.toml ./

RUN uv sync --frozen --no-install-project

# Copy source code and data
COPY src/ src/
COPY configs/ configs/
COPY README.md LICENSE ./
COPY reports/ reports/
COPY .dvc/ .dvc/

RUN uv sync --frozen

# Create output directories
RUN mkdir -p models src/stuperml/figures logs


ENTRYPOINT ["/bin/sh", "-c", "uv run dvc pull && uv run src/stuperml/train.py"]

################
# Usage example:

    # docker build -f dockerfiles/train.dockerfile . -t stuperml-train

    # # docker run -v "$(pwd)/models:/app/models" -v "$(pwd)/src/stuperml/figures:/app/src/stuperml/figures" -v "$(pwd)/logs:/app/logs" stuperml-train

# example to override parameters:

    # docker run -v $(pwd)/models:/app/models -v $(pwd)/src/stuperml/figures:/app/src/stuperml/figures stuperml-train --lr 0.00001 --epochs 15

################

