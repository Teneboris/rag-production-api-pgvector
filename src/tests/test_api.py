from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock
from llama_index.core.schema import NodeWithScore, TextNode

from src.app import main
from src.app import rag_service

#==================================================
# client
#=================================================

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(
        main.upload_sessions,
        "UPLOAD_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        main.upload_sessions,
        "initialize_upload_sessions",
        Mock(),
    )
    monkeypatch.setattr(
        main.upload_sessions,
        "cleanup_expired_uploads",
        Mock(return_value=0),
    )
    monkeypatch.setattr(main, "ProductionAgent", Mock())
    monkeypatch.setattr(main, "RAGService", Mock())

    with TestClient(main.app) as test_client:
        yield test_client
#==================================================
# Upload success
#=================================================
def test_upload_success(client, monkeypatch):
    monkeypatch.setattr(
        main.upload_sessions,
        "register_upload_session",
        Mock(),
    )
    monkeypatch.setattr(
        main,
        "ingest_uploaded_file",
        Mock(return_value=3),
    )

    response = client.post(
        "/rag/upload",
        files={
            "file": (
                "notes.txt",
                b"Temporary test document content.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["chunks_created"] == 3
    assert body["session_id"]
    assert datetime.fromisoformat(body["expires_at"]) > datetime.now(
        timezone.utc
    )

#==================================================
# rejects unsupported file
#=================================================
def test_upload_rejects_unsupported_file_type(client):
    response = client.post(
        "/rag/upload",
        files={"file": ("program.exe", b"not allowed", "application/octet-stream")},
    )

    assert response.status_code == 415

#==================================================
# Decline expired chat session
#=================================================

def test_rag_chat_rejects_expired_session(client, monkeypatch):
    monkeypatch.setattr(
        main.upload_sessions,
        "is_upload_session_active",
        Mock(return_value=False),
    )

    response = client.post(
        "/rag/chat",
        json={
            "message": "Summarize my upload",
            "session_id": "expired-session-id",
        },
    )

    assert response.status_code == 410
    
#==================================================
# Test the cleanup call for expired sessions
#=================================================
def test_cleanup_schedules_expired_sessions(monkeypatch):
    expired_rows = [
        ("expired-session", "/tmp/rag-upload-sessions/file.txt")
    ]

    query_result = Mock()
    query_result.all.return_value = expired_rows

    connection = Mock()
    connection.execute.return_value = query_result

    transaction = Mock()
    transaction.__enter__ = Mock(return_value=connection)
    transaction.__exit__ = Mock(return_value=False)

    engine = Mock()
    engine.begin.return_value = transaction

    delete_session = Mock()
    monkeypatch.setattr(
        main.upload_sessions,
        "get_engine",
        Mock(return_value=engine),
    )
    monkeypatch.setattr(
        main.upload_sessions,
        "_delete_session",
        delete_session,
    )

    deleted_count = main.upload_sessions.cleanup_expired_uploads()

    assert deleted_count == 1
    delete_session.assert_called_once_with(
        "expired-session",
        "/tmp/rag-upload-sessions/file.txt",
    )

#==================================================
# Reject uploads larger than 10 MB
#=================================================

def test_upload_rejects_file_over_limit(client, monkeypatch):
    register_mock = Mock()
    ingest_mock = Mock()

    monkeypatch.setattr(
        main.upload_sessions,
        "register_upload_session",
        register_mock,
    )
    monkeypatch.setattr(
        main,
        "ingest_uploaded_file",
        ingest_mock,
    )

    too_large = b"x" * (main.MAX_UPLOAD_BYTES + 1)

    response = client.post(
        "/rag/upload",
        files={
            "file": (
                "large.txt",
                too_large,
                "text/plain",
            )
        },
    )

    assert response.status_code == 413
    register_mock.assert_not_called()
    ingest_mock.assert_not_called()
    assert list(main.upload_sessions.UPLOAD_DIR.iterdir()) == []

#==================================================
# Session isolation during retrieval
#=================================================

def test_retrieval_filters_temporary_documents_to_requested_session(
    monkeypatch,
):
    session_id = "session-A"

    shared_node = NodeWithScore(
        node=TextNode(
            text="Shared document",
            metadata={"scope": "shared"},
        ),
        score=0.95,
    )
    session_a_node = NodeWithScore(
        node=TextNode(
            text="Private document A",
            metadata={
                "scope": "temporary",
                "session_id": session_id,
            },
        ),
        score=0.90,
    )

    shared_retriever = Mock()
    shared_retriever.retrieve.return_value = [shared_node]

    session_retriever = Mock()
    session_retriever.retrieve.return_value = [session_a_node]

    index = Mock()
    index.as_retriever.side_effect = [
        shared_retriever,
        session_retriever,
    ]

    monkeypatch.setattr(
        rag_service,
        "load_index",
        Mock(return_value=index),
    )
    monkeypatch.setattr(
        rag_service,
        "is_upload_session_active",
        Mock(return_value=True),
    )

    results = rag_service.retrieve_documents(
        "project information",
        session_id=session_id,
    )

    assert results == [shared_node, session_a_node]
    assert index.as_retriever.call_count == 2

    shared_filters = index.as_retriever.call_args_list[0].kwargs["filters"]
    temporary_filters = index.as_retriever.call_args_list[1].kwargs["filters"]

    assert [(item.key, item.value) for item in shared_filters.filters] == [
        ("scope", "shared"),
    ]
    assert {
        (item.key, item.value)
        for item in temporary_filters.filters
    } == {
        ("scope", "temporary"),
        ("session_id", session_id),
    }
    rag_service.is_upload_session_active.assert_called_once_with(session_id)





































