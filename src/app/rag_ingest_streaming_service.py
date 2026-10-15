"""Speicher-schonender Ingest fuer grosse PDF-Mengen.

Liest PDFs batchweise statt alles auf einmal, embeddet und
speichert pro Batch, und merkt sich den Fortschritt fuer Resume.
Gibt die Gesamtzahl neu gespeicherter Chunks zurueck.
"""

import json
from pathlib import Path

from llama_index.core import SimpleDirectoryReader

from src.app.config import get_settings
from src.app.monitoring import get_logger
from src.app.rag_service import (
    configure_embeddings,
    _clean_documents,
    _recursive_fallback,
    _store_nodes,
)

logger = get_logger()
settings = get_settings()

def _resolve_docs_dir() -> Path:
    if getattr(settings, "ingest_dir", None):
        docs_dir = Path(settings.ingest_dir)
    else:
        docs_dir = Path(__file__).resolve().parents[2] / "docs"

    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    return docs_dir

def _progress_path(docs_dir: Path) -> Path:
  
    override = getattr(settings, settings.ingest_progress_dir, None)
    if override:
        target = Path(override)
    else:
        project_docs = Path(__file__).resolve().parents[2] / "docs"
        target = project_docs if project_docs.is_dir() else docs_dir

    target.mkdir(parents=True, exist_ok=True)
    return target / ".ingest_progress.json"

def _load_progress(progress_file: Path) -> set[str]:
    if progress_file.is_file():
        try:
            return set(json.loads(progress_file.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Progress file unreadable; starting fresh.")
    return set()

def _save_progress(progress_file: Path, done: set[str]) -> None:
    progress_file.write_text(json.dumps(sorted(done)), encoding="utf-8")

def ingest_documents_streaming(
    *,
    fallback_chunk_size: int = 500,
    batch_size: int | None = None,
    resume: bool = True,
) -> int:
    if fallback_chunk_size < 1:
        raise ValueError("fallback_chunk_size must be at least 1.")

    batch_size = batch_size or getattr(settings, "ingest_batch_size", settings.ingest_batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    configure_embeddings()

    docs_dir = _resolve_docs_dir()
    progress_file = _progress_path(docs_dir)
    done = _load_progress(progress_file) if resume else set()

    all_pdfs = sorted(p for p in docs_dir.rglob("*.pdf") if p.is_file())
    pending = [p for p in all_pdfs if str(p) not in done]

    logger.info(
        "Streaming ingest start",
        extra={"extra_data": {
            "total_pdfs": len(all_pdfs),
            "already_done": len(done),
            "pending": len(pending),
            "batch_size": batch_size,
        }},
    )

    total_chunks = 0

    for start in range(0, len(pending), batch_size):
        batch_paths = pending[start:start + batch_size]
        try:
            documents = SimpleDirectoryReader(
                input_files=[str(p) for p in batch_paths],
            ).load_data()

            documents = _clean_documents(documents)
            if documents:
                nodes = _recursive_fallback(documents, fallback_chunk_size)
                if nodes:
                    total_chunks += _store_nodes(nodes, scope="shared")

            done.update(str(p) for p in batch_paths)
            _save_progress(progress_file, done)

            logger.info(
                "Batch ingested",
                extra={"extra_data": {
                    "batch_files": len(batch_paths),
                    "chunks_total_so_far": total_chunks,
                    "progress": f"{len(done)}/{len(all_pdfs)}",
                }},
            )
        except Exception:
            logger.exception(
                "Batch failed; progress saved, safe to resume.",
                extra={"extra_data": {
                    "batch_start": start,
                    "batch_files": [str(p) for p in batch_paths],
                }},
            )
            raise

    return total_chunks
