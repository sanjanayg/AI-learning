import threading
from typing import Dict, Any

class CacheMetricsService:
    def __init__(self):
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        
        # Latencies in milliseconds
        self.total_lookup_time = 0.0
        self.lookup_count = 0
        self.total_rag_time = 0.0
        self.rag_count = 0
        
        self.insertions = 0
        self.deletions = 0
        self.expired_removed = 0
        
        # Miss reasons
        self.miss_reasons: Dict[str, int] = {
            "below_threshold": 0,
            "metadata_mismatch": 0,
            "expired": 0,
            "old_kb_version": 0,
            "not_cacheable": 0,
        }

    def record_hit(self, lookup_time_ms: float):
        with self._lock:
            self.hits += 1
            self.total_lookup_time += lookup_time_ms
            self.lookup_count += 1

    def record_miss(self, reason: str, lookup_time_ms: float = 0.0):
        with self._lock:
            self.misses += 1
            if lookup_time_ms > 0:
                self.total_lookup_time += lookup_time_ms
                self.lookup_count += 1
            if reason in self.miss_reasons:
                self.miss_reasons[reason] += 1
            else:
                self.miss_reasons[reason] = self.miss_reasons.get(reason, 0) + 1

    def record_rag_latency(self, rag_time_ms: float):
        with self._lock:
            self.total_rag_time += rag_time_ms
            self.rag_count += 1

    def record_insertion(self):
        with self._lock:
            self.insertions += 1

    def record_deletion(self, count: int = 1):
        with self._lock:
            self.deletions += count

    def record_expired_removed(self, count: int = 1):
        with self._lock:
            self.expired_removed += count

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_requests = self.hits + self.misses
            hit_ratio = (self.hits / total_requests) if total_requests > 0 else 0.0
            avg_lookup_latency = (self.total_lookup_time / self.lookup_count) if self.lookup_count > 0 else 0.0
            avg_rag_latency = (self.total_rag_time / self.rag_count) if self.rag_count > 0 else 0.0
            
            return {
                "cache_hits": self.hits,
                "cache_misses": self.misses,
                "hit_ratio": hit_ratio,
                "average_lookup_latency_ms": avg_lookup_latency,
                "average_rag_latency_ms": avg_rag_latency,
                "cache_insertions": self.insertions,
                "cache_deletions": self.deletions,
                "expired_entries_removed": self.expired_removed,
                "miss_reasons_breakdown": dict(self.miss_reasons)
            }

    def clear(self):
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.total_lookup_time = 0.0
            self.lookup_count = 0
            self.total_rag_time = 0.0
            self.rag_count = 0
            self.insertions = 0
            self.deletions = 0
            self.expired_removed = 0
            for key in self.miss_reasons:
                self.miss_reasons[key] = 0

# Global metrics singleton
metrics_service = CacheMetricsService()
