import json
import logging
import time

from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for log aggregation."""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        # Merge any extra data attached to the record
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)


def get_logger(name: str = "production-api") -> logging.Logger:
    """Create a structured JSON logger."""

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_sum": 0.0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: bool = False,
        cache_hit: bool = False,
    ):
        self.metrics["requests_total"] += 1
        self.metrics["latency_sum"] += latency_ms
        self.metrics["latency_count"] += 1
        self.metrics["tokens_input"] += input_tokens
        self.metrics["tokens_output"] += output_tokens

        if error:
            self.metrics["errors_total"] += 1

        if cache_hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

    @property
    def get_summary(self) -> dict:
        avg_latency = (
            self.metrics["latency_sum"] / self.metrics["latency_count"]
            if self.metrics["latency_count"] > 0
            else 0
        )

        error_rate = (
            self.metrics["errors_total"] / self.metrics["requests_total"]
            if self.metrics["requests_total"] > 0
            else 0
        )

        cache_total = (
            self.metrics["cache_hits"]
            + self.metrics["cache_misses"]
        )

        cache_hit_rate = (
            self.metrics["cache_hits"] / cache_total
            if cache_total > 0
            else 0
        )

        return {
            "total_requests": self.metrics["requests_total"],
            "total_errors": self.metrics["errors_total"],
            "error_rate": f"{error_rate:.2%}",
            "avg_latency_ms": round(avg_latency, 2),
            "total_input_tokens": self.metrics["tokens_input"],
            "total_output_tokens": self.metrics["tokens_output"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
        }


class RequestTimer:
    """Context manager for timing requests."""

    def __enter__(self):
        self.start = time.perf_counter()
        self.end = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        end_time = (
            self.end
            if self.end is not None
            else time.perf_counter()
        )

        return (end_time - self.start) * 1000


def test_monitoring():
    logger = get_logger()
    metrics = MetricsCollector()

    print("=== STRUCTURED LOGGING ===")
    print()

    logger.info("Application starting")

    logger.info(
        "Processing request",
        extra={
            "extra_data": {
                "user_id": "user-123",
                "thread_id": 3,
            }
        },
    )

    logger.warning(
        "Rate limit approaching",
        extra={
            "extra_data": {
                "current_rate": 18,
                "limit_rate": 20,
            }
        },
    )

    print()
    print("=== METRICS COLLECTION ===")
    print()

    # Request 1 - cache hit
    with RequestTimer() as timer:
        time.sleep(0.1)

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=50,
        output_tokens=100,
        cache_hit=True,
    )

    print(
        f"Request 1: {timer.elapsed_ms:.1f} ms "
        f"(cache hit)"
    )

    # Request 2 - cache miss
    with RequestTimer() as timer:
        time.sleep(0.1)

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=30,
        output_tokens=80,
        cache_hit=False,
    )

    print(
        f"Request 2: {timer.elapsed_ms:.1f} ms "
        f"(cache miss)"
    )

    # Request 3 - error
    metrics.record_request(
        latency_ms=5.0,
        input_tokens=10,
        output_tokens=0,
        error=True,
        cache_hit=False,
    )

    print("Request 3: error")

    print()
    print("=== METRICS SUMMARY ===")
    print()

    print(
        json.dumps(
            metrics.get_summary,
            indent=2
        )
    )


if __name__ == "__main__":
    test_monitoring()