# ── Impact Radar — Multi-stage Docker Build ──────────────────────────
#
# === RAG Pipeline Learning: Why Docker for RAG? ===
#
# RAG systems have complex dependency chains:
#   - Python + ML libraries (numpy, torch from Chroma's deps)
#   - A vector store (Chroma with SQLite + HNSWLIB)
#   - An API framework (FastAPI + uvicorn)
#   - Data files (YAML configs, component definitions)
#
# Docker ensures "it works on my machine" becomes "it works everywhere."
# The multi-stage build keeps the final image small by separating
# build-time dependencies (compilers, pip cache) from runtime.
# ─────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies for native extensions (HNSWLIB, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY config/ ./config/
COPY data/ ./data/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Create volume mount point for Chroma persistence
# Without this, embeddings are lost when the container stops.
VOLUME ["/app/chroma_db"]

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 8000

# Health check — verifies the API is responding
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Default command: run the FastAPI server
# Use 0.0.0.0 to accept connections from outside the container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
