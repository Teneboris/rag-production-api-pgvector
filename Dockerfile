FROM python:3.14-slim

WORKDIR /app

# create non-root user to own the /app
RUN useradd --create-home appuser \
    && chown appuser:appuser /app

# install uv (fast python package manager)
RUN pip install uv

# copy dependency files first (Docker layer caching)
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser uv.lock ./

# switch to non-root user before installling deps (so .venv is ownd by appuser)
USER appuser

# install dependencies without building the local project
RUN uv sync --frozen --no-dev --no-install-project

# fix NLTK hardlink issue (st_nlink=2) by recreating the cache with fresh inodes
RUN cp -r /app/.venv/lib/python3.14/site-packages/llama_index/core/_static/nltk_cache /tmp/nltk_cache \
    && rm -rf /app/.venv/lib/python3.14/site-packages/llama_index/core/_static/nltk_cache \
    && mv /tmp/nltk_cache /app/.venv/lib/python3.14/site-packages/llama_index/core/_static/nltk_cache

# Copy application code

COPY --chown=appuser:appuser src/ src/

COPY --chown=appuser:appuser docs/ docs/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]


# Run with uvicorn
CMD ["/app/.venv/bin/uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]




















