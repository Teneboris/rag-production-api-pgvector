FROM python:3.12-slim

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

# Copy application code

COPY --chown=appuser:appuser src/ src/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1


# Run with uvicorn
CMD ["/app/.venv/bin/uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]




















