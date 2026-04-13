"""Tests for the RAG layer — vector store, embedder, and retriever.

=== RAG Pipeline Learning: Testing Without an API Key ===

A common challenge with RAG systems: the embedding API costs money and
requires credentials. Good test design lets you test retrieval logic
WITHOUT calling the real API:

  1. Use Chroma's built-in default embedding function (sentence-transformers)
     instead of OpenAI — free, local, no API key needed.
  2. Test the document preparation logic separately from embedding.
  3. Test retrieval logic with a pre-populated store.

This mirrors production practice: unit tests use mocks/local models,
integration tests use the real API, and only integration tests require
credentials.
"""

import os
import shutil
import tempfile

import pytest
from pathlib import Path

from src.graph.builder import DependencyGraph
from src.rag.vector_store import VectorStore
from src.rag.embedder import ComponentEmbedder, EmbeddingDocument
from src.rag.retriever import SemanticRetriever, ComponentMatch

# ── Fixtures ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
COMPONENTS_DIR = DATA_DIR / "components"
VARIANTS_DIR = DATA_DIR / "variants"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "model_config.yaml"


@pytest.fixture
def full_graph() -> DependencyGraph:
    return DependencyGraph(COMPONENTS_DIR, VARIANTS_DIR)


@pytest.fixture
def tmp_chroma_dir():
    """Create a temporary directory for Chroma persistence.

    === RAG Learning: Test Isolation ===

    Each test gets its own vector store directory. This prevents test
    pollution — documents added in one test don't affect another.
    Same principle as using a fresh database per test in backend testing.
    """
    tmpdir = tempfile.mkdtemp(prefix="impact_radar_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def vector_store(tmp_chroma_dir: str) -> VectorStore:
    return VectorStore(
        config_path=CONFIG_PATH,
        persist_directory=tmp_chroma_dir,
        collection_name="test_collection",
    )


@pytest.fixture
def embedder(full_graph: DependencyGraph, vector_store: VectorStore) -> ComponentEmbedder:
    return ComponentEmbedder(full_graph, vector_store, config_path=CONFIG_PATH)


@pytest.fixture
def populated_store(embedder: ComponentEmbedder, vector_store: VectorStore) -> VectorStore:
    """A vector store populated with all component documents.

    Uses Chroma's default embedding function (sentence-transformers)
    instead of OpenAI — no API key needed for tests.
    """
    embedder.embed_and_store(embedding_fn=None, reset=True)
    return vector_store


@pytest.fixture
def retriever(populated_store: VectorStore) -> SemanticRetriever:
    return SemanticRetriever(
        vector_store=populated_store,
        similarity_threshold=0.0,  # Low threshold for testing — we want results
        top_k=20,
        rerank_top_k=10,
    )


# ── Test: Document Preparation ────────────────────────────────────────

class TestDocumentPreparation:
    """Test that component data is correctly converted to embeddable documents."""

    def test_document_count(self, embedder: ComponentEmbedder):
        """Each component produces: 1 description + N module_usages + 1 api_surface + 1 summary.

        12 components with varying module counts. Let's verify the total.
        """
        docs = embedder.prepare_documents()
        assert len(docs) > 0

        # Count by type
        types = {}
        for doc in docs:
            dtype = doc.metadata["doc_type"]
            types[dtype] = types.get(dtype, 0) + 1

        # 12 descriptions, 12 summaries, 12 api_surfaces
        assert types.get("description", 0) == 12
        assert types.get("summary", 0) == 12
        assert types.get("api_surface", 0) == 12
        # Module usages = sum of all modules across components (~57)
        assert types.get("module_usage", 0) > 40

    def test_all_components_covered(self, embedder: ComponentEmbedder):
        """Every component should have at least one document."""
        docs = embedder.prepare_documents()
        component_ids = {doc.metadata["component_id"] for doc in docs}
        assert len(component_ids) == 12

    def test_document_has_required_metadata(self, embedder: ComponentEmbedder):
        """Every document must have component_id and doc_type metadata."""
        docs = embedder.prepare_documents()
        for doc in docs:
            assert "component_id" in doc.metadata
            assert "doc_type" in doc.metadata
            assert doc.metadata["doc_type"] in (
                "description", "module_usage", "api_surface", "summary"
            )

    def test_document_text_is_meaningful(self, embedder: ComponentEmbedder):
        """Document text should be natural language, not raw YAML."""
        docs = embedder.prepare_documents()
        for doc in docs:
            assert len(doc.text) > 10, f"Document {doc.doc_id} has too-short text"
            # Should NOT look like raw YAML
            assert "module_id:" not in doc.text
            assert "coupling:" not in doc.text or "Coupling:" in doc.text

    def test_description_doc_content(self, embedder: ComponentEmbedder):
        """Description docs should contain the component name and description."""
        docs = embedder.prepare_documents()
        auth_desc = next(
            d for d in docs
            if d.metadata["component_id"] == "auth_engine"
            and d.metadata["doc_type"] == "description"
        )
        assert "AuthEngine" in auth_desc.text
        assert "authentication" in auth_desc.text.lower()

    def test_module_usage_doc_content(self, embedder: ComponentEmbedder):
        """Module usage docs should mention the component, module, and usage."""
        docs = embedder.prepare_documents()
        crypto_doc = next(
            d for d in docs
            if d.metadata.get("module_id") == "crypto_utils"
            and d.metadata["component_id"] == "auth_engine"
        )
        assert "AuthEngine" in crypto_doc.text
        assert "crypto_utils" in crypto_doc.text
        assert "coupling" in crypto_doc.text.lower()

    def test_document_ids_are_unique(self, embedder: ComponentEmbedder):
        """Every document must have a unique ID."""
        docs = embedder.prepare_documents()
        ids = [doc.doc_id for doc in docs]
        assert len(ids) == len(set(ids)), "Duplicate document IDs found"

    def test_document_stats(self, embedder: ComponentEmbedder):
        stats = embedder.get_document_stats()
        assert stats["total_documents"] > 80
        assert stats["components_covered"] == 12
        assert stats["total_characters"] > 5000
        assert "description" in stats["doc_type_counts"]
        assert "module_usage" in stats["doc_type_counts"]


# ── Test: Vector Store ────────────────────────────────────────────────

class TestVectorStore:
    """Test the Chroma vector store wrapper."""

    def test_empty_store(self, vector_store: VectorStore):
        assert vector_store.count() == 0

    def test_add_and_count(self, vector_store: VectorStore):
        vector_store.add_documents(
            documents=["Test document about authentication"],
            metadatas=[{"component_id": "test", "doc_type": "description"}],
            ids=["test_1"],
        )
        assert vector_store.count() == 1

    def test_add_and_query(self, vector_store: VectorStore):
        """Add a document and retrieve it via semantic search."""
        vector_store.add_documents(
            documents=[
                "Handles user authentication and password management",
                "Processes payments and generates invoices for billing",
            ],
            metadatas=[
                {"component_id": "auth", "doc_type": "description"},
                {"component_id": "billing", "doc_type": "description"},
            ],
            ids=["auth_desc", "billing_desc"],
        )

        results = vector_store.query(query_text="login and passwords", n_results=2)
        assert len(results) > 0
        # The auth document should be more relevant to "login and passwords"
        assert results[0].component_id == "auth"

    def test_upsert_idempotent(self, vector_store: VectorStore):
        """Adding the same document ID twice should update, not duplicate."""
        for _ in range(2):
            vector_store.add_documents(
                documents=["Test document"],
                metadatas=[{"component_id": "test", "doc_type": "description"}],
                ids=["test_1"],
            )
        assert vector_store.count() == 1

    def test_reset(self, vector_store: VectorStore):
        vector_store.add_documents(
            documents=["Test"],
            metadatas=[{"component_id": "test", "doc_type": "description"}],
            ids=["test_1"],
        )
        assert vector_store.count() == 1
        vector_store.reset()
        assert vector_store.count() == 0

    def test_query_with_metadata_filter(self, vector_store: VectorStore):
        """Test filtered vector search — a key RAG pattern."""
        vector_store.add_documents(
            documents=[
                "Authentication handles login",
                "Auth module uses crypto for hashing",
            ],
            metadatas=[
                {"component_id": "auth", "doc_type": "description"},
                {"component_id": "auth", "doc_type": "module_usage"},
            ],
            ids=["auth_desc", "auth_module"],
        )

        results = vector_store.query(
            query_text="authentication",
            n_results=5,
            where={"doc_type": "module_usage"},
        )
        # Should only return module_usage docs
        for r in results:
            assert r.doc_type == "module_usage"

    def test_query_empty_store_returns_empty(self, vector_store: VectorStore):
        results = vector_store.query(query_text="anything", n_results=5)
        assert results == []

    def test_get_by_component(self, vector_store: VectorStore):
        vector_store.add_documents(
            documents=["Auth description", "Auth module usage"],
            metadatas=[
                {"component_id": "auth", "doc_type": "description"},
                {"component_id": "auth", "doc_type": "module_usage"},
            ],
            ids=["auth_1", "auth_2"],
        )
        results = vector_store.get_by_component("auth")
        assert len(results) == 2

    def test_similarity_score_range(self, vector_store: VectorStore):
        """Similarity scores should be between 0 and 1."""
        vector_store.add_documents(
            documents=["password hashing and encryption"],
            metadatas=[{"component_id": "test", "doc_type": "description"}],
            ids=["test_1"],
        )
        results = vector_store.query(query_text="password hashing", n_results=1)
        assert len(results) == 1
        assert 0.0 <= results[0].similarity <= 1.0


# ── Test: Full Embedding Pipeline ─────────────────────────────────────

class TestEmbeddingPipeline:
    """Test the complete embed → store → query pipeline.

    Uses Chroma's default embedding function (no API key needed).
    """

    def test_embed_and_store_count(self, embedder: ComponentEmbedder):
        """embed_and_store should return the number of documents stored."""
        count = embedder.embed_and_store(embedding_fn=None, reset=True)
        assert count > 80  # 12 components × ~7 docs each

    def test_store_populated_after_embedding(self, populated_store: VectorStore):
        """After embedding, the store should have documents."""
        assert populated_store.count() > 80

    def test_query_returns_relevant_results(self, populated_store: VectorStore):
        """Querying for 'authentication' should return AuthEngine-related docs."""
        results = populated_store.query(
            query_text="user authentication and login sessions",
            n_results=5,
        )
        assert len(results) > 0
        component_ids = {r.component_id for r in results}
        assert "auth_engine" in component_ids

    def test_query_billing_returns_billing_docs(self, populated_store: VectorStore):
        results = populated_store.query(
            query_text="payment processing invoices subscription billing",
            n_results=5,
        )
        component_ids = {r.component_id for r in results}
        assert "billing_core" in component_ids

    def test_module_query_finds_shared_module(self, populated_store: VectorStore):
        """Querying for a module's usage should find components using it."""
        results = populated_store.query(
            query_text="database connection pool management",
            n_results=10,
            where={"doc_type": "module_usage"},
        )
        assert len(results) > 0
        # db_connector is used by billing_core, audit_logger, search_service, report_generator
        component_ids = {r.component_id for r in results}
        assert len(component_ids) >= 2  # At least 2 components using db-related modules


# ── Test: Semantic Retriever ──────────────────────────────────────────

class TestSemanticRetriever:
    """Test the retriever's aggregation, filtering, and ranking logic."""

    def test_find_related_returns_component_matches(self, retriever: SemanticRetriever):
        results = retriever.find_related_components(
            query="authentication and user sessions"
        )
        assert len(results) > 0
        assert all(isinstance(r, ComponentMatch) for r in results)

    def test_results_sorted_by_score(self, retriever: SemanticRetriever):
        results = retriever.find_related_components(
            query="encryption and cryptographic operations"
        )
        scores = [r.best_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_exclude_component(self, retriever: SemanticRetriever):
        """Excluded components should not appear in results."""
        results = retriever.find_related_components(
            query="authentication and user sessions",
            exclude_component_ids=["auth_engine"],
        )
        component_ids = {r.component_id for r in results}
        assert "auth_engine" not in component_ids

    def test_find_related_to_component(self, retriever: SemanticRetriever, full_graph: DependencyGraph):
        """find_related_to_component should find semantically similar components."""
        auth = full_graph.get_component("auth_engine")
        results = retriever.find_related_to_component(
            component_id="auth_engine",
            component_description=auth.description,
        )
        # auth_engine should be excluded from its own results
        component_ids = {r.component_id for r in results}
        assert "auth_engine" not in component_ids
        # Should find security-related components
        assert len(results) > 0

    def test_component_match_properties(self, retriever: SemanticRetriever):
        results = retriever.find_related_components(
            query="payment processing"
        )
        if results:
            match = results[0]
            assert match.component_id != ""
            assert match.best_score > 0
            assert match.match_count > 0
            assert match.avg_score > 0

    def test_rerank_limits_results(self, retriever: SemanticRetriever):
        """Retriever should return at most rerank_top_k results."""
        results = retriever.find_related_components(
            query="general platform operations"
        )
        assert len(results) <= retriever._rerank_top_k

    def test_search_modules(self, retriever: SemanticRetriever):
        """search_modules should only return module_usage documents."""
        results = retriever.search_modules(
            query="Redis session persistence"
        )
        for r in results:
            assert r.doc_type == "module_usage"

    def test_doc_type_filter(self, retriever: SemanticRetriever):
        """doc_type_filter should scope results to a specific type."""
        results = retriever.find_related_components(
            query="authentication",
            doc_type_filter="description",
        )
        for match in results:
            for doc in match.matching_docs:
                assert doc.doc_type == "description"
