from types import SimpleNamespace
from llama_index.core import Document
from unittest.mock import Mock, MagicMock
from llama_index.core.schema import NodeWithScore, TextNode

from src.app import rag_service

class TestRAGService:
    
    def test_configure_embeddings(self, monkeypatch):
        
        monkeypatch.setattr(
        rag_service,
        "get_settings",
        Mock(
            return_value=SimpleNamespace(
                embeddings_model="chroma/all-minilm-l6-v2-f32:latest",
                ollama_base_url="http://localhost:11434",
                )
            ),
        )
        
        mock_embedding_model = Mock()
        mock_embedding_model.get_text_embedding.return_value = [0.1, 0.2, 0.3, 0.4]

        monkeypatch.setattr(
            rag_service,
            "OllamaEmbedding",
            Mock(return_value=mock_embedding_model),
        )

        fake_settings_obj = SimpleNamespace()
        monkeypatch.setattr(
            rag_service,
            "LlamaSettings",
            fake_settings_obj
        )
        
        result = rag_service.configure_embeddings()
        
        assert result == 4
        rag_service.OllamaEmbedding.assert_called_once_with(
            model_name="chroma/all-minilm-l6-v2-f32:latest",
            base_url="http://localhost:11434",
        )
        
    def test_ingest_documents(self, monkeypatch, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        fake_module_path = tmp_path / "src" / "app" / "rag_service.py"
        monkeypatch.setattr(rag_service, "__file__", str(fake_module_path))

        source_documents = [
            Document(text="First source document"),
            Document(text="Second source document"),
        ]
        monkeypatch.setattr(
            rag_service,
            "SimpleDirectoryReader",
            Mock(return_value=Mock(load_data=Mock(return_value=source_documents))),
        )

        mock_store = object()
        monkeypatch.setattr(
            rag_service,
            "create_vector_store",
            Mock(return_value=mock_store),
        )

        mock_nodes = [
            TextNode(text="Chunk one"),
            TextNode(text="Chunk two"),
            TextNode(text="Chunk three"),
        ]
        monkeypatch.setattr(
            rag_service,
            "_recursive_fallback",
            Mock(return_value=mock_nodes),
        )

        mock_storage_context = object()
        storage_context_mock = Mock(
            from_defaults=Mock(return_value=mock_storage_context)
        )
        monkeypatch.setattr(
            rag_service,
            "StorageContext",
            storage_context_mock,
        )

        vector_index_mock = Mock()
        monkeypatch.setattr(
            rag_service,
            "VectorStoreIndex",
            vector_index_mock,
        )

        result = rag_service.ingest_documents(use_semantic=False)

        assert result == 3
        rag_service._recursive_fallback.assert_called_once_with(
            source_documents,
            500,
        )
        storage_context_mock.from_defaults.assert_called_once_with(
            vector_store=mock_store,
        )
        vector_index_mock.assert_called_once_with(
            mock_nodes,
            storage_context=mock_storage_context,
        )
            
    def test_load_index(self, monkeypatch):
        rag_service.load_index.cache_clear()
        
        expected_index = object()
        
        monkeypatch.setattr(
            rag_service,
            "configure_embeddings",
            Mock(return_value=1536)
        )
        
        monkeypatch.setattr(
            rag_service,
            "create_vector_store",
            Mock(return_value=1536)
        )
        
        monkeypatch.setattr(
            rag_service.VectorStoreIndex,
            "from_vector_store",
            Mock(return_value=expected_index)
        )
        
        result = rag_service.load_index()
        
        assert result is expected_index
        rag_service.create_vector_store.assert_called_once()
        rag_service.VectorStoreIndex.from_vector_store.assert_called_once()

    def test_retrieve_documents(self, monkeypatch):
        query = "What is the purpose of this project?"
        expected_document = NodeWithScore(
            node=TextNode(text="This project is designed to demonstrate the capabilities of a RAG system."),
            score=0.92,
        )
        
        monkeypatch.setattr(
            rag_service,
            "retrieve_documents",
            Mock(return_value={"documents": expected_document}),
        )
        
        state = {"query": query}
        result = rag_service.retrieve_documents(state)
        
        rag_service.retrieve_documents.assert_called_once_with(state)
        assert result == {"documents": expected_document}

    def test_format_context(self):
        documents = [
            NodeWithScore(
                node=TextNode(
                    text="This project is designed to demonstrate the capabilities of a RAG system.",
                    metadata={"file_name": "project.md"},
                ),
                score=0.92,
            )
        ]

        result = rag_service.format_context(documents)

        assert "This project is designed to demonstrate the capabilities of a RAG system." in result
        assert "Source: project.md" in result
        assert "Retrieval score: 0.920" in result
        

    def test_grade_document(self, monkeypatch):
        state = {
            "query": "What is the purpose of this project?",
            "documents": [
                NodeWithScore(
                    node=TextNode(text="This project is designed to demonstrate the capabilities of a RAG system."),
                    score=0.9,
                ),
                NodeWithScore(
                    node=TextNode(text="This is unrelated company information."),
                    score=0.1,
                )
            ],
            "threshold": 0.7,
        }
        
        monkeypatch.setattr(
            rag_service,
            "get_settings",
            Mock(
                return_value=SimpleNamespace(
                    primary_model="gemma4:cloud",
                    ollama_base_url="http://localhost:11434",
                    temprature=0,
                )
            )
        )
        
        fake_llm = Mock()
        fake_llm.with_structured_output = fake_llm
        fake_llm.invoke.side_effect = [
            rag_service.RelevanceGrade(
                relevant=True,
                reasoning="This document clearly answers the question.",
                llm_score=0.9,
            ),
            
            rag_service.RelevanceGrade(
                relevant=False,
                reasoning="This document is unrelated to the question.",
                llm_score=0.1,
            ),
        ]

        fake_prompt = Mock()
        fake_prompt.__or__=Mock(return_value=fake_llm)
        monkeypatch.setattr(
            rag_service,
            "grading_prompt",
            fake_prompt,
        )
        result = rag_service.grade_document(state)

        assert result["relevance_score"] == 0.5
        assert len(result["relevant_documents"]) == 1
        assert result["document_grades"][0]["relevance"] is True
        assert result["document_grades"][0]["llm_score"] == 0.9

    def test_rewrite_query(self, monkeypatch):
        state = {
            "query": "What is this project?",
        }
        
        monkeypatch.setattr(
            rag_service,
            "get_settings",
            Mock(
                return_value=SimpleNamespace(
                    primary_model="gemma4:cloud",
                    ollama_base_url="http://localhost:11434",
                    temprature=0,
                )
            )
        )
        fake_llm = Mock()
        fake_llm.invoke.return_value = Mock(content="What is the purpose of this project?")
        
        fake_prompt = Mock()
        fake_prompt.__or__=Mock(return_value=fake_llm)
        
        monkeypatch.setattr(
            rag_service,
            "ChatPromptTemplate",
            Mock(return_value=fake_prompt),
        )
        
        result = rag_service.rewrite_query(state)
        
        assert result["rewritten_query"] == "What is the purpose of this project?"
        assert result["retry_count"] == 1

    def test_generate_answer(self, monkeypatch):
        state = {
            "query": "What is the purpose of this project?",
            "relevant_documents": [
                NodeWithScore(
                    node=TextNode(text="This project is designed to demonstrate the capabilities of a RAG system."),
                    score=0.9,
                )
            ],
        }
        
        monkeypatch.setattr(
        rag_service,
        "get_settings",
        Mock(
            return_value=SimpleNamespace(
                primary_model="gemma4:cloud",
                ollama_base_url="http://localhost:11434",
            )
        ),
    )

        monkeypatch.setattr(
            rag_service,
            "format_context",
            Mock(return_value={rag_service.format_context(state.get("relevant_documents"))}),
    )

        fake_chain = Mock()
        fake_chain.invoke.return_value = SimpleNamespace(
            content="The project demonstrates a RAG system."
        )

        fake_prompt = MagicMock()
        fake_prompt.__or__ = Mock(return_value=fake_chain)

        monkeypatch.setattr(
            rag_service.ChatPromptTemplate,
            "from_messages",
            Mock(return_value=fake_prompt),
        )
        
        result = rag_service.generate_answer(state)
        
        generated_answer = "The project demonstrates a RAG system."
        
        assert result["generated_answer"] == generated_answer
        

    def test_generate_fallback(self, monkeypatch):
        fallback_answer_mock = "This is a fallback answer."
        
        monkeypatch.setattr(
            rag_service,
            "generate_fallback",
            Mock(return_value={"generated_fallback": fallback_answer_mock}),
        )
        
        result = rag_service.generate_fallback()

        assert result["generated_fallback"] == fallback_answer_mock

    def test_should_retry_or_generate(self):
        state = {
            "retry_count": 3,
            "max_retries": 3,
            "document": [],
            "relevance_score": 0,
        }
        
        result = rag_service.should_retry_or_generate(state)
        
        assert result == "fallback"

    def test_should_retry_or_generate_generate_path(self):
        state = {
            "retry_count": 0,
            "max_retries": 3,
            "documents": [{"node": "dummy"}],
            "relevance_score": 0.8,
        }

        result = rag_service.should_retry_or_generate(state)

        assert result == "generate"

    def test_should_retry_or_generate_retry_path(self):
        state = {
            "retry_count": 1,
            "max_retries": 3,
            "documents": [],
            "relevance_score": 0.2,
        }

        result = rag_service.should_retry_or_generate(state)

        assert result == "rewrite"
        
    def test_build_agentic_rag_graph_compiles(self):
        graph = rag_service.build_agentic_rag_graph()

        assert graph is not None



















































