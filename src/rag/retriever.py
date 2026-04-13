"""Semantic retriever — queries the vector store to find related components.

=== RAG Pipeline Learning: Retrieval Strategy ===

This module implements the retrieval logic that sits between the vector store
(raw similarity search) and the impact analyzer (which needs structured results).

The retriever adds intelligence on top of raw vector search:
  1. QUERY EXPANSION — turn a component ID into a rich natural language query
  2. RESULT FILTERING — apply similarity threshold to remove weak matches
  3. DEDUPLICATION — group results by component (multiple docs per component)
  4. RESULT RANKING — combine multiple signals (similarity, criticality, coupling)

=== Two-Stage Retrieval ===

Our config implements a two-stage retrieval pattern:
  Stage 1: Retrieve top_k=10 candidates from vector search (cast wide net)
  Stage 2: Rerank to top rerank_top_k=5 (refine for precision)

This is a standard production pattern. Stage 1 optimizes for RECALL (don't miss
anything relevant). Stage 2 optimizes for PRECISION (only keep the best matches).
In a full system, Stage 2 would use a cross-encoder reranker (like Cohere Rerank).
Here we use a simpler signal-based reranking for educational clarity.

=== How Retriever Complements Graph Traversal ===

The graph traverser finds structurally connected components (via shared modules).
The retriever finds semantically similar components (via text similarity).

Example of what the retriever catches that the graph misses:
  - AuthEngine and APIGateway are graph-connected (share access_control)
  - AuthEngine and EncryptionService are SEMANTICALLY similar (both deal
    with security, keys, cryptography) — even though they only share
    crypto_utils, their descriptions overlap in ways the graph doesn't capture

The impact analyzer merges both result sets for comprehensive coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.rag.vector_store import VectorStore, RetrievalResult


@dataclass
class ComponentMatch:
    """A semantically matched component with aggregated similarity scores.

    When multiple documents from the same component match a query, we
    aggregate them into a single ComponentMatch with the best score and
    all matching document details.

    === RAG Learning: Aggregation Matters ===

    Raw vector search returns document-level matches, but the consumer
    (impact analyzer, LLM) thinks in terms of components. Aggregation
    bridges this gap. The 'best_score' represents the strongest semantic
    signal, while 'matching_docs' preserves the evidence for the LLM prompt.
    """

    component_id: str
    component_name: str
    best_score: float  # Highest similarity among matching docs
    criticality: str
    matching_docs: list[RetrievalResult] = field(default_factory=list)

    @property
    def match_count(self) -> int:
        """How many documents from this component matched."""
        return len(self.matching_docs)

    @property
    def avg_score(self) -> float:
        """Average similarity across all matching documents."""
        if not self.matching_docs:
            return 0.0
        return sum(r.similarity for r in self.matching_docs) / len(self.matching_docs)


class SemanticRetriever:
    """Retrieves semantically related components from the vector store.

    Wraps VectorStore with domain-specific query logic: query expansion,
    result aggregation by component, threshold filtering, and reranking.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        similarity_threshold: float = 0.7,
        top_k: int = 10,
        rerank_top_k: int = 5,
    ) -> None:
        """Initialize the retriever with search parameters.

        Args:
            vector_store: The Chroma vector store instance.
            similarity_threshold: Minimum similarity to include (0-1).
            top_k: Number of raw results from vector search (Stage 1).
            rerank_top_k: Number of final results after reranking (Stage 2).

        === RAG Learning: Tuning Retrieval ===

        These three parameters are the primary tuning knobs:
          - similarity_threshold too HIGH → misses relevant but differently-worded matches
          - similarity_threshold too LOW → includes noise that confuses the LLM
          - top_k too LOW → misses relevant results in Stage 1 (can't recover in Stage 2)
          - rerank_top_k too HIGH → sends too much context to the LLM (token waste)

        Start with the defaults from config and adjust based on retrieval quality
        evaluation — a step we'd add in a production RAG system.
        """
        self._store = vector_store
        self._similarity_threshold = similarity_threshold
        self._top_k = top_k
        self._rerank_top_k = rerank_top_k

    @classmethod
    def from_config(
        cls,
        vector_store: VectorStore,
        config: dict[str, Any] | None = None,
    ) -> SemanticRetriever:
        """Create a retriever with parameters from model_config.yaml."""
        retrieval_config = (config or {}).get("retrieval", {})
        return cls(
            vector_store=vector_store,
            similarity_threshold=retrieval_config.get("similarity_threshold", 0.7),
            top_k=retrieval_config.get("top_k", 10),
            rerank_top_k=retrieval_config.get("rerank_top_k", 5),
        )

    def find_related_components(
        self,
        query: str,
        exclude_component_ids: list[str] | None = None,
        doc_type_filter: str | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[ComponentMatch]:
        """Find components semantically related to a natural language query.

        Args:
            query: Natural language query (e.g., "session management and authentication").
            exclude_component_ids: Component IDs to exclude from results (typically
                                  the changed components themselves).
            doc_type_filter: Only search specific doc types ("description",
                           "module_usage", "api_surface", "summary").
            query_embedding: Pre-computed query embedding for consistency with
                           document embeddings. If None, Chroma auto-embeds.

        Returns:
            List of ComponentMatch objects, sorted by best_score descending.

        === RAG Learning: Exclude the Query Source ===

        When analyzing the impact of changing AuthEngine, we exclude AuthEngine
        from retrieval results. We already KNOW AuthEngine is affected — we want
        to find OTHER components that are semantically related. This is similar
        to how a search engine excludes the page you're currently viewing from
        "related pages" suggestions.
        """
        # Build metadata filter
        where: dict[str, Any] | None = None
        if doc_type_filter:
            where = {"doc_type": doc_type_filter}

        # Query the vector store
        raw_results = self._store.query(
            query_text=query if query_embedding is None else None,
            query_embedding=query_embedding,
            n_results=self._top_k,
            where=where,
        )

        # Filter by similarity threshold and excluded components
        exclude_set = set(exclude_component_ids or [])
        filtered: list[RetrievalResult] = []
        for result in raw_results:
            if result.similarity < self._similarity_threshold:
                continue
            if result.component_id in exclude_set:
                continue
            filtered.append(result)

        # Aggregate by component
        component_matches = self._aggregate_by_component(filtered)

        # Sort by best_score descending and apply rerank limit
        component_matches.sort(key=lambda m: m.best_score, reverse=True)
        return component_matches[: self._rerank_top_k]

    def find_related_to_component(
        self,
        component_id: str,
        component_description: str,
        exclude_component_ids: list[str] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[ComponentMatch]:
        """Find components semantically related to a specific component.

        This is a convenience method that uses the component's description
        as the query. Useful in the impact analyzer when enriching graph
        traversal results with semantic matches.

        === RAG Learning: Component-as-Query ===

        Instead of a user typing a natural language query, we use the
        component's own description as the query. This is a common RAG
        pattern called "query by example" — find documents similar to this
        document. It's how recommendation systems work: "show me items
        similar to this item."
        """
        exclude = list(set((exclude_component_ids or []) + [component_id]))
        return self.find_related_components(
            query=component_description,
            exclude_component_ids=exclude,
            query_embedding=query_embedding,
        )

    def _aggregate_by_component(
        self, results: list[RetrievalResult]
    ) -> list[ComponentMatch]:
        """Group document-level results into component-level matches.

        === RAG Learning: From Documents to Entities ===

        Vector stores operate on documents (text chunks). But our domain
        model operates on components. This aggregation step bridges the gap:
          - Multiple docs from AuthEngine → one ComponentMatch for AuthEngine
          - best_score = max similarity across all matching docs
          - matching_docs preserved for LLM context building

        This is a general pattern in entity-centric RAG systems (medical records,
        legal documents, code components) where the retrieval unit (document)
        differs from the reasoning unit (entity).
        """
        component_map: dict[str, ComponentMatch] = {}

        for result in results:
            comp_id = result.component_id
            if not comp_id:
                continue

            if comp_id not in component_map:
                component_map[comp_id] = ComponentMatch(
                    component_id=comp_id,
                    component_name=result.metadata.get("component_name", comp_id),
                    best_score=result.similarity,
                    criticality=result.metadata.get("criticality", "medium"),
                    matching_docs=[result],
                )
            else:
                match = component_map[comp_id]
                match.matching_docs.append(result)
                if result.similarity > match.best_score:
                    match.best_score = result.similarity

        return list(component_map.values())

    def search_modules(
        self,
        query: str,
        query_embedding: list[float] | None = None,
    ) -> list[RetrievalResult]:
        """Search specifically within module usage documents.

        Returns raw RetrievalResult objects (not aggregated by component)
        because the caller typically wants module-level granularity.

        Useful for queries like "what uses Redis?" or "password hashing".
        """
        return self._store.query(
            query_text=query if query_embedding is None else None,
            query_embedding=query_embedding,
            n_results=self._top_k,
            where={"doc_type": "module_usage"},
        )
