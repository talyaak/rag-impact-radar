"""Chroma vector store wrapper for component embeddings.

=== RAG Pipeline Learning: The Vector Store ===

A vector store is a specialized database optimized for similarity search
over high-dimensional vectors (embeddings). Think of it as:

  Traditional DB:  "SELECT * FROM components WHERE name = 'AuthEngine'"
  Vector store:    "Find the 5 components most semantically similar to
                    'user session management and authentication'"

This is the core infrastructure that enables the "R" (Retrieval) in RAG.
When a user asks about the impact of changing a component, the vector store
finds semantically related components that the graph traversal might miss.

=== Why Chroma? ===

Chroma is chosen for several educational reasons:
  1. Zero infrastructure — runs in-process with local persistence
  2. Native Python — no Docker/server setup for development
  3. Built-in embedding function support (though we use OpenAI's API directly)
  4. Simple API that maps cleanly to RAG concepts: add, query, delete

In production, you'd likely use Pinecone, Weaviate, or pgvector for
scalability and operational features. But the API patterns are identical.

=== Document Model ===

Each "document" in our vector store represents one aspect of a component:
  - Component description (what it does)
  - Module usage entries (how it uses each shared module)
  - API surface (what endpoints it exposes)

We store these as separate documents with metadata linking back to the
component ID. This granularity means a query about "session management"
can match the specific module usage entry rather than the entire component
description — more precise retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
import yaml


@dataclass
class RetrievalResult:
    """A single result from semantic search.

    Contains the matched document text, its metadata (component_id, doc_type),
    the similarity score, and the original document ID for deduplication.

    === RAG Learning: Metadata Is Essential ===

    Raw vector search returns text chunks, but downstream consumers need to
    know WHERE each chunk came from. The metadata (component_id, doc_type,
    module_id) lets us:
      1. Group results by component (for the impact report)
      2. Link semantic matches back to graph nodes (for hybrid retrieval)
      3. Filter by type (e.g., only search module usage descriptions)
    """

    document: str
    metadata: dict[str, Any]
    distance: float  # Lower = more similar for cosine distance
    doc_id: str

    @property
    def similarity(self) -> float:
        """Convert Chroma distance to similarity score (0-1).

        Chroma uses cosine DISTANCE (1 - cosine_similarity), so we
        convert back to similarity for intuitive scoring.
        """
        return 1.0 - self.distance

    @property
    def component_id(self) -> str:
        return self.metadata.get("component_id", "")

    @property
    def doc_type(self) -> str:
        return self.metadata.get("doc_type", "")


class VectorStore:
    """Manages a Chroma collection for component embeddings.

    Provides methods to add documents, query by text or vector, and
    manage the collection lifecycle. All config is driven by model_config.yaml.

    === RAG Learning: Collection = Index ===

    A Chroma "collection" is analogous to a database table or a search index.
    All documents in a collection share the same embedding space and can be
    compared via similarity search. We use a single collection for all
    component documents — this lets cross-component similarity search work
    naturally.
    """

    def __init__(
        self,
        config_path: str | Path = "config/model_config.yaml",
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Initialize the vector store from config.

        Args:
            config_path: Path to model_config.yaml.
            persist_directory: Override the persist directory from config.
            collection_name: Override the collection name from config.
        """
        self._config = self._load_config(config_path)
        vs_config = self._config.get("vector_store", {})

        self._persist_dir = persist_directory or vs_config.get(
            "persist_directory", "./chroma_db"
        )
        self._collection_name = collection_name or vs_config.get(
            "collection_name", "impact_radar_components"
        )
        distance_metric = vs_config.get("distance_metric", "cosine")

        # Initialize Chroma client with local persistence
        self._client = chromadb.PersistentClient(path=self._persist_dir)

        # Get or create the collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": distance_metric},
        )

    @staticmethod
    def _load_config(config_path: str | Path) -> dict[str, Any]:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of text strings to store.
            metadatas: Parallel list of metadata dicts per document.
            ids: Parallel list of unique IDs per document.
            embeddings: Pre-computed embeddings. If None, Chroma will use
                       its default embedding function (Sentence Transformers).

        === RAG Learning: Pre-computed vs. Auto Embeddings ===

        You can let Chroma auto-embed using its built-in model, or provide
        your own embeddings (from OpenAI, Cohere, etc.). We use pre-computed
        OpenAI embeddings because:
          1. OpenAI's text-embedding-3-small is higher quality for our domain
          2. We control the embedding model version (reproducibility)
          3. We can embed in batches for efficiency
          4. Query embeddings must match document embeddings — using the same
             model for both is mandatory
        """
        kwargs: dict[str, Any] = {
            "documents": documents,
            "metadatas": metadatas,
            "ids": ids,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        self._collection.upsert(**kwargs)

    def query(
        self,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search for similar documents.

        Args:
            query_text: Natural language query (Chroma auto-embeds this).
            query_embedding: Pre-computed query vector (use for consistency
                           with pre-computed document embeddings).
            n_results: Number of results to return (this is top_k).
            where: Metadata filter (e.g., {"doc_type": "module_usage"}).
            where_document: Document content filter.

        Returns:
            List of RetrievalResult sorted by similarity (best first).

        === RAG Learning: Query-Time Filtering ===

        The `where` parameter is powerful: it lets you combine vector
        similarity with metadata constraints. For example:
          - query="session management", where={"doc_type": "module_usage"}
            → finds module usage entries about sessions, skipping descriptions
          - query="encryption", where={"component_id": {"$ne": "encryption_service"}}
            → finds encryption-related text in OTHER components

        This is called "filtered vector search" — a standard RAG pattern
        for scoping retrieval to relevant document types.
        """
        kwargs: dict[str, Any] = {
            "n_results": min(n_results, self._collection.count() or n_results),
        }
        if query_text is not None:
            kwargs["query_texts"] = [query_text]
        elif query_embedding is not None:
            kwargs["query_embeddings"] = [query_embedding]
        else:
            raise ValueError("Either query_text or query_embedding must be provided")

        if where is not None:
            kwargs["where"] = where
        if where_document is not None:
            kwargs["where_document"] = where_document

        # Handle empty collection
        if self._collection.count() == 0:
            return []

        results = self._collection.query(**kwargs)

        # Parse Chroma's nested result format into RetrievalResult objects
        retrieval_results: list[RetrievalResult] = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                retrieval_results.append(
                    RetrievalResult(
                        document=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        distance=results["distances"][0][i] if results["distances"] else 0.0,
                        doc_id=doc_id,
                    )
                )

        return retrieval_results

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection. Use for rebuilding embeddings."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._config.get("vector_store", {}).get("distance_metric", "cosine")},
        )

    def get_by_component(self, component_id: str) -> list[RetrievalResult]:
        """Retrieve all documents for a specific component.

        Useful for inspecting what's stored about a component, or for
        building component-specific context for the LLM.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.get(
            where={"component_id": component_id},
            include=["documents", "metadatas"],
        )

        retrieval_results: list[RetrievalResult] = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                retrieval_results.append(
                    RetrievalResult(
                        document=results["documents"][i] if results["documents"] else "",
                        metadata=results["metadatas"][i] if results["metadatas"] else {},
                        distance=0.0,  # Not a similarity query
                        doc_id=doc_id,
                    )
                )

        return retrieval_results

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)
