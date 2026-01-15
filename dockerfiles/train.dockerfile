# Base image with Python 3.13 on Debian Bookworm slim
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY uv.lock pyproject.toml ./
COPY configs/ configs/

RUN uv sync --frozen --no-install-project

# Copy source code and data
COPY src/ src/
COPY data/ data/
COPY README.md LICENSE ./

RUN uv sync --frozen

# Create output directories
RUN mkdir -p models src/stuperml/figures


ENTRYPOINT ["uv", "run", "src/stuperml/train.py"]


################
# Usage example:

    # docker build -f dockerfiles/train.dockerfile . -t stuperml-train

    # docker run -v $(pwd)/models:/app/models -v $(pwd)/src/stuperml/figures:/app/src/stuperml/figures stuperml-train

# example to override parameters:

    # docker run -v $(pwd)/models:/app/models -v $(pwd)/src/stuperml/figures:/app/src/stuperml/figures stuperml-train --lr 0.00001 --epochs 15

################

