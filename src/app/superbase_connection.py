
from llama_index.vector_stores.postgres import PGVectorStore

from src.app.rag_service import create_vector_store

class SuperbaseConnection():
    def connect_to_supabase(self) -> PGVectorStore:
        """
        Connect to Supabase and return a PGVectorStore instance.
        """
        vectorstore = create_vector_store()
        return vectorstore















































