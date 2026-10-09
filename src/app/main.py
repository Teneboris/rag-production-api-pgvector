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


from datetime import datetime, timedelta, timezone
import time
import asyncio
import secrets
from pathlib import Path
from contextlib import asynccontextmanager, suppress
from fastapi import (
    FastAPI,
    Request,
    HTTPException,
    UploadFile,
    File
)
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
    RelevanceResponse,
    RAGChatRequest,
    UploadSessionResponse,
    OpenAIChatMessage,
    OpenAIChatRequest
)

from src.app.security import SecurityPipeline
from src.app.cache import ResponseCache
from src.app.monitoring import get_logger, MetricsCollector, RequestTimer
from src.app.agent import ProductionAgent
from src.app.rag_service import RAGService, ingest_uploaded_file
from src.app import upload_sessions

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
rag_agent = None

MAX_UPLOAD_BYTES = 10 * 1024 * 1024 # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize components on startup and clean up on shutdown.
    """

    global security, cache, metrics, agent, rag_agent

    await asyncio.to_thread(upload_sessions.initialize_upload_sessions)
    await asyncio.to_thread(upload_sessions.cleanup_expired_uploads)
    cleanup_task = asyncio.create_task(cleanup_upload_sessions_loop())

    try:
        security = SecurityPipeline()
        cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
        metrics = MetricsCollector()
        agent = ProductionAgent()
        rag_agent = RAGService()

        
        logger.info(
            "All components initialized. Ready to serve requests."
        )

        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
            
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

    logger.info(
        "Shutting down...",
        extra={
            "extra_data": metrics.get_summary
        },
    )

async def cleanup_upload_sessions_loop() -> None:
    while True:
        try:
            deleted = await asyncio.to_thread(
                upload_sessions.cleanup_expired_uploads
            )
            if deleted:
                logger.info("Cleaned up %s expired upload sessions", deleted)
        except Exception:
            logger.exception("Upload-session cleanup failed")

        await asyncio.sleep(20)
        
# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Production LangGraph API",
    description=(
        "A production-ready chat and RAG system API with security, "
        "caching, and observability."
    ),
    version="1.0.1",
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
# rag upload
# ============================================================


@app.post("/rag/upload", response_model=UploadSessionResponse)
@limiter.limit("5/minute")
async def upload_rag_file(
    request: Request,
    file: UploadFile = File(...),
):
    original_name = (file.filename or "upload").replace("\\", "/")
    filename = Path(original_name).name
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Upload PDF, TXT oder Markdown.")

    session_id = secrets.token_urlsafe(32)
    upload_path = upload_sessions.UPLOAD_DIR / f"{session_id}{suffix}"
    upload_sessions.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    size = 0
    registered = False

    try:
        with upload_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Datei überschreitet 10 MB.")
                destination.write(chunk)

        if size == 0:
            raise HTTPException(400, "Die hochgeladene Datei ist leer.")

        expires_at = datetime.now(timezone.utc) + upload_sessions.UPLOAD_TTL

        await asyncio.to_thread(
            upload_sessions.register_upload_session,
            session_id,
            filename,
            upload_path,
            expires_at,
        )
        registered = True

        chunks_created = await asyncio.to_thread(
            ingest_uploaded_file,
            str(upload_path),
            filename,
            session_id,
            expires_at,
        )

        return UploadSessionResponse(
            session_id=session_id,
            filename=filename,
            chunks_created=chunks_created,
            expires_at=expires_at,
        )

    except Exception as exc:
        if registered:
            try:
                await asyncio.to_thread(
                    upload_sessions.delete_upload_session,
                    session_id,
                )
            except Exception:
                logger.exception("Failed to roll back upload session")

        upload_path.unlink(missing_ok=True)

        if isinstance(exc, HTTPException):
            raise

        logger.exception("Upload ingestion failed")
        raise HTTPException(500, "Datei konnte nicht verarbeitet werden.") from exc

    finally:
        await file.close()

# ============================================================
# rag Chat Endpoint
#=============================================================

@app.post(
    "/rag/chat",
    response_model=RelevanceResponse,
)
@limiter.limit(settings.rate_limit)
@traceable(name="rag_chat_endpoint")
async def rag_chat(
    request: Request,
    body: RAGChatRequest,
):
    """
    RAG chat endpoint.

    Flow:
    1. Security check
    2. RAG-specific cache lookup
    3. Invoke the RAG graph with the query
    4. Select the generated answer or fallback response
    5. Output validation
    6. Cache store
    7. Metrics and response
    """
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
                latency_ms=timer.elapsed_ms,
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
            
        if body.session_id:
            active = await asyncio.to_thread(
                upload_sessions.is_upload_session_active,
                body.session_id,
            )
                    
            if not active:
                raise HTTPException(
                    status_code=410,
                    detail="Upload-Session ist abgelaufen oder unbekannt."
                )
    # ----------------------------------------------------
    # Step 2: RAG-specific cache lookup
    # ----------------------------------------------------

        rag_cache_key = f"rag:{body.session_id or 'shared'}:{cleaned_message}"
        
        cached_response = cache.get(
            rag_cache_key
        )
        
        if cached_response is not None:
            
            metrics.record_request(
                latency_ms=timer.elapsed_ms,
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

            return RelevanceResponse(
                query=cleaned_message,
                response=cached_response,
                relevant_documents=[],
                document_grades=[],
                relevance_score=None,
                threshold=0.5,
                thread_id=body.thread_id or "default",
                model_used="cache",
                cache=True,
                processing_time_ms=0,
            )
    # --------------------------------------------------------
    # Step 3: Invoke the RAG graph with the query
    # --------------------------------------------------------
        try:
            
            rag_result = await asyncio.to_thread(
                rag_agent.invoke,
                cleaned_message,
                body.session_id,
            )
        except Exception as exc:
            logger.exception(
                "RAG graph invocation failed",
                extra={"extra_data": {"error": str(exc)}},
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
                detail="An error occurred while processing your RAG request.",
            )
            
        response_text = rag_result["response"]
        model_used = rag_result["model_used"]
        
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
            rag_cache_key,
            validated_response,
        )

        # ----------------------------------------------------
        # Step 6: Metrics and response
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
    # --------------------------------------------------------
    # Step 8: Select the generated answer or fallback response
    # --------------------------------------------------------
        return RelevanceResponse(
            query=cleaned_message,
            response=validated_response,
            relevant_documents=rag_result["relevant_documents"],
            document_grades=rag_result["document_grades"],
            relevance_score=rag_result["relevance_score"],
            threshold=rag_result["threshold"],
            thread_id=body.thread_id or "default",
            model_used=model_used,
            cache=False,
            processing_time_ms=round(timer.elapsed_ms, 2),
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
# OpenAI-compatible endpoints (Open WebUI integration)
# ============================================================

@app.get("/v1/models")
async def list_models():
    """Minimal OpenAI-compatible model list for Open WebUI."""
    return {
        "object": "list",
        "data": [
            {
                "id": "rag-gemma",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rag-production",
            }
        ],
    }

@app.post("/v1/chat/completions")
@limiter.limit(settings.rate_limit)
async def openai_chat_completions(
    request: Request,
    body: OpenAIChatRequest,
):
    """
    OpenAI-compatible chat endpoint that bridges to the RAG graph.
    Non-streaming only.
    """
    with RequestTimer() as timer:
        # letzte User-Nachricht herausziehen
        user_message = next(
            (m.content for m in reversed(body.messages) if m.role == "user"),
            None,
        )
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message provided.")

        # Step 1: Security-Check
        is_allowed, cleaned_message, notes = security.check_input(user_message)
        if not is_allowed:
            logger.warning(
                "Request blocked by security",
                extra={"extra_data": {"reason": notes}},
            )
            metrics.record_request(
                latency_ms=timer.elapsed_ms,
                input_tokens=0,
                output_tokens=0,
                error=True,
                cache_hit=False,
            )
            raise HTTPException(
                status_code=400,
                detail="Your message was blocked by our security filters.",
            )

        # Step 2: RAG aufrufen
        try:
            rag_result = await asyncio.to_thread(
                rag_agent.invoke,
                cleaned_message,
            )
        except Exception as exc:
            logger.exception(
                "RAG graph invocation failed (openai endpoint)",
                extra={"extra_data": {"error": str(exc)}},
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
                detail="An error occurred while processing your request.",
            )

        # Step 4: Output validieren
        answer, _ = security.check_output(rag_result["response"])
        model_used = rag_result.get("model_used", settings.primary_model)

        # Step 5: Metrics
        input_tokens = int(len(cleaned_message.split()) * 1.3)
        output_tokens = int(len(answer.split()) * 1.3)
        metrics.record_request(
            latency_ms=timer.elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=False,
        )

        # Step 6: OpenAI-kompatible Response
        return {
            "id": f"chatcmpl-{secrets.token_hex(12)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_used,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }


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