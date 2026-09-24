"""
Production-Ready FastAPI + LangGraph Application

wire together:
- Security pipeline (input sanitization, PII maskin)
- Response caching
- Rate limiting (slowapi)
- LangGraph agent (with retries + fallback)
- Structured logging + metrics
- LangSmith tracing
- Health checks
"""


import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from langsmith import traceable
from dotenv import load_dotenv

from src.app.config import get_settings
from src.app.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MetricsResponse,
)

from src.app.security import SecurityPipeline
from src.app.cache import ResponseCache
from src.app.monitoring import get_logger, MetricsCollector, RequestTimer
from src.app.agent import ProductionAgent


load_dotenv()

logger = get_logger()
settings = get_settings()

limiter = Limiter(
    key_func=get_remote_address
)


security = None
cache = None
metrics = None
agent = None


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize components on startup and clean up on shutdown.
    """

    global security, cache, metrics, agent

    logger.info(
        "Starting production API...",
        extra={
            "extra_data": {
                "environment": settings.app_env,
                "primary_model": settings.primary_model,
                "tracing_enabled": settings.langchain_tracing_v2,
            }
        },
    )

    security = SecurityPipeline()

    cache = ResponseCache(
        ttl_seconds=settings.cache_ttl_seconds
    )

    metrics = MetricsCollector()

    agent = ProductionAgent()

    logger.info(
        "All components initialized. Ready to serve requests."
    )

    yield

    logger.info(
        "Shutting down...",
        extra={
            "extra_data": metrics.get_summary
        },
    )


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Production LangGraph API",
    description=(
        "A production-ready chat API with security, "
        "caching, and observability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter


# ============================================================
# Exception Handlers
# ============================================================

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(
    request: Request,
    exc: RateLimitExceeded,
):
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please try again later."
        },
    )


# ============================================================
# Chat Endpoint
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
@limiter.limit(settings.rate_limit)
@traceable(name="chat_endpoint")
async def chat(
    request: Request,
    body: ChatRequest,
):
    """
    Main chat endpoint.

    Flow:
    1. Security check
    2. Cache lookup
    3. LangGraph agent invoke
    4. Output validation
    5. Cache store
    6. Metrics and response
    """
    print(f"this just to test the cleaned message")
    with RequestTimer() as timer:

        security_notes = []

        # ----------------------------------------------------
        # Step 1: Security Check
        # ----------------------------------------------------

        is_allowed, cleaned_message, notes = (
            security.check_input(body.message)
        )

        security_notes.extend(notes)

        if not is_allowed:

            logger.warning(
                "Request blocked by security",
                extra={
                    "extra_data": {
                        "reason": notes,
                        "thread_id": body.thread_id,
                    }
                },
            )

            metrics.record_request(
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                error=True,
                cache_hit=False,
            )

            raise HTTPException(
                status_code=400,
                detail=(
                    "Your message was blocked by "
                    "our security filters."
                ),
            )

        # ----------------------------------------------------
        # Step 2: Cache Lookup
        # ----------------------------------------------------

        cached_response = cache.get(
            cleaned_message
        )
    
        if cached_response is not None:
            
            metrics.record_request(
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                cache_hit=True,
            )

            logger.info(
                "Cache hit",
                extra={
                    "extra_data": {
                        "thread_id": body.thread_id,
                    }
                },
            )

            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cache=True,
                processing_time_ms=0
            )

        # ----------------------------------------------------
        # Step 3: LangGraph Agent
        # ----------------------------------------------------

        try:
            result = agent.invoke(
                cleaned_message
            )

        except Exception as e:

            logger.error(
                "Agent invocation failed",
                extra={
                    "extra_data": {
                        "thread_id": body.thread_id,
                        "error": str(e),
                    }
                },
            )

            metrics.record_request(
                latency_ms=timer.elapsed_ms,
                input_tokens=0,
                output_tokens=0,
                error=True,
                cache_hit=False,
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "An error occurred while processing "
                    "your request."
                ),
            )

        response_text = result["response"]
        model_used = result["model_used"]

        # ----------------------------------------------------
        # Step 4: Output Validation
        # ----------------------------------------------------

        validated_response, output_warnings = (
            security.check_output(
                response_text
            )
        )

        security_notes.extend(
            output_warnings
        )

        # ----------------------------------------------------
        # Step 5: Cache Store
        # ----------------------------------------------------

        cache.set(
            cleaned_message,
            validated_response,
        )

        # ----------------------------------------------------
        # Step 6: Metrics
        # ----------------------------------------------------

        input_tokens = int(
            len(cleaned_message.split()) * 1.3 # * 1.3 estimates that each word uses about 1.3 tokens.
        )

        output_tokens = int(
            len(validated_response.split()) * 1.3
        )

        metrics.record_request(
            latency_ms=timer.elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=False,
        )

        if security_notes:
            logger.info(
                "Security notes",
                extra={
                    "extra_data": {
                        "notes": security_notes,
                        "thread_id": body.thread_id,
                    }
                },
            )

        return ChatResponse(
            response=validated_response,
            thread_id=body.thread_id,
            model_used=model_used,
            cache=False,
            processing_time_ms=round(
                timer.elapsed_ms,
                2,
            ),
        )


# ============================================================
# Health Endpoint
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health():
    """
    Health check for Docker/Kubernetes.
    """

    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
        "metrics": metrics is not None,
    }

    all_healthy = all(
        checks.values()
    )

    return HealthResponse(
        status=(
            "healthy"
            if all_healthy
            else "degraded"
        ),
        environment=settings.app_env,
        checks=checks,
    )

# ============================================================
# Metrics Endpoint
# ============================================================

@app.get(
    "/metrics",
    response_model=MetricsResponse,
)
async def get_metrics():
    """
    Metrics for monitoring dashboards.
    """

    summary = metrics.get_summary

    return MetricsResponse(
        **summary
    )


# ============================================================
# Cache Stats Endpoint
# ============================================================

@app.get("/cache/stats")
async def cache_stats():
    """
    Cache performance statistics.
    """

    return cache.stats