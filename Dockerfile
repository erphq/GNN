# Slim CPU image. CUDA images are much larger and most users running this
# in a container are doing batch / CI work where CPU is fine.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps:
# - graphviz/libgomp: required by some PyG / sklearn paths
# - git: pip needs it for any VCS-style installs (none today, but cheap)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU torch first, then the rest of the stack — the CPU index
# avoids pulling the multi-GB CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt pyproject.toml README.md LICENSE ./
RUN pip install -r requirements.txt

# Source code goes after the deps so a code-only edit doesn't bust the
# pip cache.
COPY gnn_cli/ gnn_cli/
COPY models/ models/
COPY modules/ modules/
COPY visualization/ visualization/
COPY main.py ./
RUN pip install -e .

# Sample data so `gnn run` works on a fresh `docker run` without volume mounts.
COPY input/ input/

# `gnn` is the canonical entry point, but allow `docker run … python main.py`
# fallthrough by keeping an empty CMD.
ENTRYPOINT ["gnn"]
CMD ["--help"]
