import pytest
from llama_index.vector_stores.postgres import PGVectorStore
from app.superbase_connection import SuperbaseConnection

class TestSupabaseConnection:
    
    def setup_method(self):
        self.connection = SuperbaseConnection()
    
    def test_supabase_connection(self):
        
        try:
            vector_store = self.connection.connect_to_supabase()
            assert isinstance(vector_store, PGVectorStore)
        except Exception as e:
            pytest.fail(f"Failed to connect to Supabase: {e}")


































