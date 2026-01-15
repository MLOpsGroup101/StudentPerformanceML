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
RUN mkdir -p src/stuperml/figures

ENTRYPOINT ["uv", "run", "src/stuperml/evaluate.py"]


################
# Usage example:

    # docker build -f dockerfiles/evaluate.dockerfile . -t stuperml-evaluate

    ## evaluate the saved model (model.pth): 
    # docker run -v $(pwd)/models:/app/models -v $(pwd)/src/stuperml/figures:/app/src/stuperml/figures stuperml-evaluate --model-checkpoint models/model.pth

################