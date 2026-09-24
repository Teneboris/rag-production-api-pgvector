"""
Test for the caching layer.
Fast, deterministic, no external dependencies.
"""
import time
from app.cache import ResponseCache

class TestResponseCache:
    """Test the response cache."""

    def setup_method(self):
        self.cache = ResponseCache()

    def test_cache_miss_returns_none(self):
        assert self.cache.get("unknown query") is None

    def test_cache_hit_returns_response(self):
        self.cache.set("What is python?", "A Programming language.")
        result = self.cache.get("What is Python?")
        assert result == "A Programming language."

    def test_ttl_expiration(self):
        self.cache = ResponseCache(ttl_seconds=1)
        self.cache.set("query", "response")
        assert self.cache.get("query") == "response"
        time.sleep(1.5)
        assert self.cache.get("query") is None

    def test_stats_tracking(self):
        self.cache.get("miss1")
        self.cache.get("miss2")
        self.cache.set("hit", "value")
        self.cache.get("hit")

        stats = self.cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["cached_entries"] == 1

