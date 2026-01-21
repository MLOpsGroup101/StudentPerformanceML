# Base image with Python 3.13 on Debian Bookworm slim
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Install ONLY frontend dependencies
# We use '--system' to install directly into python, skipping the need for a venv activation
# We do NOT use 'uv sync' here to avoid installing PyTorch (saving ~2GB of space)
RUN uv pip install --system streamlit requests

COPY frontend/ frontend/


EXPOSE 8080

# Run Streamlit
ENTRYPOINT ["/bin/sh", "-c", "uv run streamlit run frontend/ui.py --server.port=8080 --server.address=0.0.0.0"]