import logging
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
import tempfile

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

from src.app.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)
UPLOAD_TTL = timedelta(minutes=10)
UPLOAD_DIR = Path(tempfile.gettempdir())/"rag-upload-sessions"

@lru_cache(maxsize=1)
def get_engine() -> Engine:
    if not settings.supabase_database_url:
        raise ValueError("SUPABASE_DATABASE_URL is not configured.")

    url = make_url(settings.supabase_database_url)
    url = url.set(drivername="postgresql+psycopg2")
    return create_engine(url, pool_pre_ping=True)

def _vector_table_name() -> str:
    name = settings.collection_name
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("Invalid PGVector table name.")
    return f'"{name}"'

def initialize_upload_sessions() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with get_engine().begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS rag_upload_sessions (
                session_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                expires_at TIMESTAMPTZ NOT NULL
            )
        """))
        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS
            ix_rag_upload_sessions_expires_at
            ON rag_upload_sessions (expires_at)
        """))

def register_upload_session(
    session_id: str,
    filename: str,
    file_path: Path,
    expires_at: datetime,
):
    with get_engine().begin() as connection:
        connection.execute(
            text("""
                INSERT INTO rag_upload_sessions
                    (session_id, filename, file_path, expires_at)
                VALUES
                    (:session_id, :filename, :file_path, :expires_at)
            """),
            {
                "session_id": session_id,
                "filename": filename,
                "file_path": str(file_path),
                "expires_at": expires_at,
            },
        )

def is_upload_session_active(session_id: str) -> bool:
    with get_engine().begin() as connection:
        return connection.execute(
            text("""
                SELECT 1
                FROM rag_upload_sessions
                WHERE session_id = :session_id
                AND expires_at > now()
            """),
            {"session_id": session_id},
        ).first() is not None

def _delete_session(session_id: str, file_path: str) -> None:
    upload_path = Path(file_path).resolve()
    upload_root = UPLOAD_DIR.resolve()

    if not upload_path.is_relative_to(upload_root):
        raise ValueError("Upload path is outside the upload directory.")

    table_name = settings.collection_name
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError("Invalid PGVector table name.")

    with get_engine().begin() as connection:
        table_exists = connection.execute(
            text("SELECT to_regclass(:table_name)"),
            {"table_name": table_name},
        ).scalar()

        if table_exists is not None:
            connection.execute(
                text(f"""
                    DELETE FROM "{table_name}"
                    WHERE metadata_ ->> 'session_id' = :session_id
                """),
                {"session_id": session_id},
            )

        connection.execute(
            text("""
                DELETE FROM rag_upload_sessions
                WHERE session_id = :session_id
            """),
            {"session_id": session_id},
        )

    upload_path.unlink(missing_ok=True)

def delete_upload_session(session_id: str,) -> None:
    with get_engine().begin() as connection:
        row = connection.execute(
            text("""
                SELECT file_path
                FROM rag_upload_sessions
                WHERE session_id = :session_id
            """),
            {"session_id": session_id},
        ).first()

    if row is not None:
        _delete_session(session_id, row.file_path)

def cleanup_expired_uploads() -> int:
    with get_engine().begin() as connection:
        expired = connection.execute(
            text("""
                SELECT session_id, file_path
                FROM rag_upload_sessions
                WHERE expires_at <= now()
            """)
        ).all()
        
    deleted = 0
    for session_id, file_path in expired:
        try:
            _delete_session(session_id, file_path)
            deleted += 1
        except Exception:
            logger.exception(
                "Failed to clean expired upload session %s",
                session_id,
        )
    return deleted
























