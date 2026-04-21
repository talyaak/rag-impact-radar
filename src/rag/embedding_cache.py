"""Content-hash embedding cache — skip re-embedding identical text.

Keeps a persistent dict of `sha256(text) -> embedding vector`. When the
embedder is asked to produce embeddings for N texts, any text whose hash
is already in the cache is served locally for $0. Only the uncached
texts are dispatched to the embedding API. The cache is persisted to a
JSON file between runs.

Why this matters: Impact Radar V2 can be pointed at the same repository
multiple times (first for gap detection, later for embedding, later again
after refinement). Without a cache, every ingestion re-pays the embedding
bill. With the cache, the bill is paid once per unique description.

Thread-safety: the cache is single-process, single-thread. If you need
concurrent writers, wrap `get`/`put` in a lock at the call site.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _hash_text(text: str, model: str) -> str:
    """Hash the text together with the model name.

    Different embedding models produce different vectors for the same
    text, so the cache key MUST include the model identifier to avoid
    returning stale vectors from a previously configured model.
    """
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_requests if self.total_requests else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 4),
        }


class EmbeddingCache:
    """Persistent LRU-ish cache mapping content hashes to embedding vectors.

    Storage format (JSON):
        {
            "version": 1,
            "model": "text-embedding-3-small",
            "entries": {
                "<sha256-hex>": [0.12, -0.04, ...],
                ...
            }
        }

    The `model` field is informational; actual cache keys include the
    model name so different-model entries never collide.
    """

    VERSION = 1

    def __init__(
        self,
        cache_path: str | Path,
        max_entries: int = 100_000,
        enabled: bool = True,
    ) -> None:
        self._path = Path(cache_path)
        self._max_entries = max_entries
        self._enabled = enabled
        self._entries: "OrderedDict[str, list[float]]" = OrderedDict()
        self._stats = CacheStats()
        self._dirty = False

        if self._enabled:
            self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                raw = json.load(f)
            entries = raw.get("entries", {}) if isinstance(raw, dict) else {}
            for k, v in entries.items():
                if isinstance(v, list):
                    self._entries[k] = [float(x) for x in v]
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to load embedding cache at %s: %s", self._path, e)
            self._entries = OrderedDict()

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
            self._stats.evictions += 1

    def get(self, text: str, model: str) -> list[float] | None:
        """Return a cached embedding for (text, model), or None on miss."""
        if not self._enabled:
            return None
        key = _hash_text(text, model)
        hit = self._entries.get(key)
        if hit is not None:
            self._entries.move_to_end(key)  # LRU: mark as recently used
            self._stats.hits += 1
            return list(hit)
        self._stats.misses += 1
        return None

    def put(self, text: str, model: str, embedding: list[float]) -> None:
        """Store an embedding. Marks cache as dirty; call flush() to persist."""
        if not self._enabled:
            return
        key = _hash_text(text, model)
        self._entries[key] = list(embedding)
        self._entries.move_to_end(key)
        self._evict_if_needed()
        self._dirty = True

    def flush(self) -> None:
        """Persist the cache to disk if there are unsaved changes."""
        if not self._enabled or not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = {
            "version": self.VERSION,
            "entries": self._entries,
        }
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, self._path)
        self._dirty = False

    def clear(self) -> None:
        self._entries.clear()
        self._stats = CacheStats()
        self._dirty = True

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def enabled(self) -> bool:
        return self._enabled
