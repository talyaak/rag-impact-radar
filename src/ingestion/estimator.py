"""Dry-run cost estimator — projects token counts and $ spend before ingestion.

Runs the free parts of the pipeline (scan + parse + gap detection) against
a repository path, measures the content that *would* be sent to external
APIs, and multiplies by configured per-token prices. No OpenAI calls are
made. Returns a structured projection the caller can inspect (or display)
before deciding to run the paid pipeline.

Price inputs live in config/model_config.yaml under `pricing:`. Update
them when provider prices change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.graph.builder import DependencyGraph
from src.rag.embedder import ComponentEmbedder
from src.rag.vector_store import VectorStore
from src.ingestion.engine import IngestionEngine
from src.gap_analysis.detector import GapDetector


def _approx_tokens(text: str, chars_per_token: int) -> int:
    """Approximate token count from character length."""
    if not text:
        return 0
    return max(1, len(text) // max(1, chars_per_token))


@dataclass
class EmbeddingProjection:
    document_count: int = 0
    total_chars: int = 0
    approx_tokens: int = 0
    batches: int = 0
    cache_hits_assumed: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_count": self.document_count,
            "total_chars": self.total_chars,
            "approx_tokens": self.approx_tokens,
            "batches": self.batches,
            "cache_hits_assumed": self.cache_hits_assumed,
            "cost_usd": round(self.cost_usd, 4),
        }


@dataclass
class GapSuggestionProjection:
    gap_count: int = 0
    llm_calls: int = 0
    approx_input_tokens: int = 0
    approx_output_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_count": self.gap_count,
            "llm_calls": self.llm_calls,
            "approx_input_tokens": self.approx_input_tokens,
            "approx_output_tokens": self.approx_output_tokens,
            "cost_usd": round(self.cost_usd, 4),
        }


@dataclass
class CostProjection:
    repo_path: str
    components_found: int = 0
    modules_found: int = 0
    embeddings: EmbeddingProjection = field(default_factory=EmbeddingProjection)
    gap_suggestions: GapSuggestionProjection = field(default_factory=GapSuggestionProjection)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return round(self.embeddings.cost_usd + self.gap_suggestions.cost_usd, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_path": self.repo_path,
            "components_found": self.components_found,
            "modules_found": self.modules_found,
            "embeddings": self.embeddings.to_dict(),
            "gap_suggestions": self.gap_suggestions.to_dict(),
            "total_cost_usd": self.total_cost_usd,
            "warnings": list(self.warnings),
        }


class CostEstimator:
    """Projects embedding + gap-suggestion cost for ingesting a repo."""

    def __init__(
        self,
        config: dict[str, Any],
        embedding_cache: Any | None = None,
    ) -> None:
        self._config = config
        self._cache = embedding_cache

        pricing = config.get("pricing", {})
        self._embed_price = float(pricing.get("embedding_per_million_tokens", 0.02))
        self._llm_input_price = float(pricing.get("llm_input_per_million_tokens", 2.50))
        self._llm_output_price = float(pricing.get("llm_output_per_million_tokens", 10.00))
        self._chars_per_token = int(pricing.get("chars_per_token", 4))

        embed_config = config.get("embedding", {})
        self._embed_batch_size = int(embed_config.get("batch_size", 100))
        self._embed_model = embed_config.get("model", "text-embedding-3-small")

        gap_config = config.get("gap_analysis", {})
        self._suggestion_batch_size = max(1, int(gap_config.get("suggestion_batch_size", 1)))
        self._enable_suggestions = gap_config.get("enable_llm_suggestions", True)

    def estimate(
        self,
        repo_path: str | Path,
        include_embeddings: bool = True,
        include_gap_suggestions: bool = True,
    ) -> CostProjection:
        """Run the free pipeline stages and project costs.

        Args:
            repo_path: Path to the repository to analyze.
            include_embeddings: Whether to project embedding cost.
            include_gap_suggestions: Whether to project gap-suggestion cost.
        """
        projection = CostProjection(repo_path=str(repo_path))

        # Stage 1: scan + parse + graph build (always free, no API calls).
        engine = IngestionEngine()
        ingest_result = engine.ingest(repo_path=repo_path, embed=False)
        graph = ingest_result.graph
        projection.components_found = ingest_result.components_ingested
        projection.modules_found = ingest_result.modules_ingested
        projection.warnings.extend(ingest_result.warnings)

        # Stage 2: project embedding cost (no API calls).
        if include_embeddings:
            projection.embeddings = self._project_embeddings(graph)

        # Stage 3: detect gaps (free) + project suggestion cost.
        if include_gap_suggestions and self._enable_suggestions:
            projection.gap_suggestions = self._project_gap_suggestions(graph)

        return projection

    def _project_embeddings(self, graph: DependencyGraph) -> EmbeddingProjection:
        """Count documents an embedder would produce; subtract cache hits."""
        proj = EmbeddingProjection()

        # Build the same docs the embedder would, but without calling the API.
        store_stub = _NullVectorStore()
        embedder = ComponentEmbedder(graph, store_stub)
        documents = embedder.prepare_documents()

        proj.document_count = len(documents)
        total_tokens = 0
        hits = 0
        for doc in documents:
            total_tokens += _approx_tokens(doc.text, self._chars_per_token)
            proj.total_chars += len(doc.text)
            if self._cache is not None and self._cache.get(doc.text, self._embed_model) is not None:
                hits += 1

        proj.approx_tokens = total_tokens
        proj.cache_hits_assumed = hits

        paid_docs = max(0, proj.document_count - hits)
        paid_tokens = max(
            0,
            total_tokens - sum(
                _approx_tokens(doc.text, self._chars_per_token)
                for doc in documents
                if self._cache is not None
                and self._cache.get(doc.text, self._embed_model) is not None
            ),
        )
        proj.batches = (paid_docs + self._embed_batch_size - 1) // self._embed_batch_size if paid_docs else 0
        proj.cost_usd = (paid_tokens / 1_000_000) * self._embed_price

        return proj

    def _project_gap_suggestions(self, graph: DependencyGraph) -> GapSuggestionProjection:
        """Count gaps and project how many LLM calls + tokens the session would cost."""
        proj = GapSuggestionProjection()

        detector = GapDetector(graph)
        report = detector.detect()
        unresolved = report.unresolved
        proj.gap_count = len(unresolved)

        if not unresolved:
            return proj

        # Graph context is sent with every (batched) prompt.
        components = graph.get_components()
        graph_context_chars = 200 + 80 * len(components)

        calls = (proj.gap_count + self._suggestion_batch_size - 1) // self._suggestion_batch_size
        per_gap_chars = 180  # heuristic: average gap description + type/severity
        input_chars_total = calls * graph_context_chars + proj.gap_count * per_gap_chars

        # Output: ~120 chars per suggestion (1-2 sentences)
        output_chars_total = proj.gap_count * 120

        proj.llm_calls = calls
        proj.approx_input_tokens = _approx_tokens(" " * input_chars_total, self._chars_per_token)
        proj.approx_output_tokens = _approx_tokens(" " * output_chars_total, self._chars_per_token)
        proj.cost_usd = (
            (proj.approx_input_tokens / 1_000_000) * self._llm_input_price
            + (proj.approx_output_tokens / 1_000_000) * self._llm_output_price
        )
        return proj


class _NullVectorStore:
    """Stub that satisfies the ComponentEmbedder constructor without touching Chroma."""

    def reset(self) -> None:
        return None

    def add_documents(self, **_: Any) -> None:
        return None
