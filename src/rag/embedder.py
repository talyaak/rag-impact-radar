"""Component embedder — converts YAML component data into vector store documents.

=== RAG Pipeline Learning: Document Preparation ===

This is arguably the most important step in a RAG pipeline. The quality of
your retrieval is bounded by the quality of your documents. Bad documents →
bad retrieval → bad generation, no matter how good your LLM is.

Key decisions made here:

1. WHAT to embed: We don't embed raw YAML. We create natural language
   "documents" from the structured data. The LLM embedding model was trained
   on natural language, so "AuthEngine uses crypto_utils for password hashing"
   embeds much better than "module_id: crypto_utils, coupling: tight".

2. GRANULARITY: We create multiple documents per component:
   - One for the component description (high-level purpose)
   - One per module usage (specific dependency relationships)
   This granularity means a search for "password hashing" returns the specific
   module usage, not just "AuthEngine" generally. More precise retrieval =
   more relevant context for the LLM.

3. METADATA: Every document carries component_id, doc_type, and other metadata.
   This lets us filter at query time and link results back to graph nodes for
   hybrid retrieval.

4. CHUNKING: For our small dataset, each document is already within the
   chunk_size limit. In a larger system, you'd split long descriptions into
   overlapping chunks — the chunk_size and chunk_overlap config parameters
   control this. We include the chunking logic for educational completeness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.graph.builder import DependencyGraph
from src.rag.vector_store import VectorStore


@dataclass
class EmbeddingDocument:
    """A prepared document ready for embedding and storage.

    === RAG Learning: The Document Contract ===

    This is the "unit of retrieval" ��� when the vector store returns a match,
    it returns one of these. The text must be semantically meaningful on its
    own (the embedding model sees no metadata). The metadata must be rich
    enough to link the result back to the knowledge graph and component model.
    """

    doc_id: str
    text: str
    metadata: dict[str, Any]


class ComponentEmbedder:
    """Converts component data into embeddable documents and stores them.

    This class bridges the structured world (YAML/graph) and the semantic
    world (embeddings/vector store). It transforms structured component
    definitions into natural language documents suitable for embedding.
    """

    def __init__(
        self,
        dep_graph: DependencyGraph,
        vector_store: VectorStore,
        config_path: str | Path = "config/model_config.yaml",
    ) -> None:
        self._graph = dep_graph
        self._store = vector_store
        self._config = self._load_config(config_path)
        self._chunk_size = self._config.get("chunking", {}).get("chunk_size", 512)
        self._chunk_overlap = self._config.get("chunking", {}).get("chunk_overlap", 64)

    @staticmethod
    def _load_config(config_path: str | Path) -> dict[str, Any]:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def prepare_documents(self) -> list[EmbeddingDocument]:
        """Convert all components into embeddable documents.

        Creates multiple document types per component for granular retrieval:

        1. 'description' — The component's high-level purpose and capabilities.
           Good for broad queries like "which component handles payments?"

        2. 'module_usage' — One per module, describing how the component uses it.
           Good for specific queries like "what uses Redis for caching?"

        3. 'api_surface' — The component's API endpoints as a document.
           Good for queries like "which endpoints handle authentication?"

        4. 'summary' — A combined summary with component name, team, criticality,
           and all module names. Acts as a catch-all for general similarity.

        === RAG Learning: Why Multiple Documents Per Component? ===

        Embedding a single giant document per component means "crypto_utils for
        password hashing" gets averaged into the same vector as "publishes auth
        events." The embedding loses specificity. By splitting into focused
        documents, each embedding captures a tight semantic cluster. A query about
        "password hashing" matches the crypto_utils module_usage doc directly,
        with high similarity, instead of vaguely matching the entire AuthEngine.
        """
        documents: list[EmbeddingDocument] = []

        for comp_id, comp_data in self._graph.get_components().items():
            comp = comp_data  # ComponentData object

            # 1. Component description document
            desc_text = (
                f"{comp.name} ({comp.id}): {comp.description.strip()}\n"
                f"Team: {comp.team_owner}. Criticality: {comp.criticality}."
            )
            documents.append(EmbeddingDocument(
                doc_id=f"{comp_id}__description",
                text=desc_text,
                metadata={
                    "component_id": comp_id,
                    "component_name": comp.name,
                    "doc_type": "description",
                    "criticality": comp.criticality,
                    "team_owner": comp.team_owner,
                },
            ))

            # 2. Module usage documents (one per module)
            for mod in comp.modules:
                mod_text = (
                    f"{comp.name} uses module '{mod['module_id']}': "
                    f"{mod.get('usage', 'No description available')}. "
                    f"Coupling: {mod.get('coupling', 'unknown')}."
                )
                documents.append(EmbeddingDocument(
                    doc_id=f"{comp_id}__module__{mod['module_id']}",
                    text=mod_text,
                    metadata={
                        "component_id": comp_id,
                        "component_name": comp.name,
                        "doc_type": "module_usage",
                        "module_id": mod["module_id"],
                        "coupling": mod.get("coupling", "unknown"),
                        "criticality": comp.criticality,
                    },
                ))

            # 3. API surface document
            if comp.api_surface:
                api_text = (
                    f"{comp.name} API endpoints:\n"
                    + "\n".join(f"  - {ep}" for ep in comp.api_surface)
                )
                documents.append(EmbeddingDocument(
                    doc_id=f"{comp_id}__api_surface",
                    text=api_text,
                    metadata={
                        "component_id": comp_id,
                        "component_name": comp.name,
                        "doc_type": "api_surface",
                        "criticality": comp.criticality,
                    },
                ))

            # 4. Combined summary document
            module_names = [m["module_id"] for m in comp.modules]
            summary_text = (
                f"Component: {comp.name} (ID: {comp_id})\n"
                f"Purpose: {comp.description.strip()}\n"
                f"Team: {comp.team_owner}\n"
                f"Criticality: {comp.criticality}\n"
                f"Shared modules: {', '.join(module_names)}\n"
                f"API endpoints: {len(comp.api_surface)}"
            )
            documents.append(EmbeddingDocument(
                doc_id=f"{comp_id}__summary",
                text=summary_text,
                metadata={
                    "component_id": comp_id,
                    "component_name": comp.name,
                    "doc_type": "summary",
                    "criticality": comp.criticality,
                    "team_owner": comp.team_owner,
                    "module_count": len(module_names),
                },
            ))

        return documents

    def embed_and_store(
        self,
        embedding_fn: Any = None,
        reset: bool = False,
    ) -> int:
        """Prepare documents, optionally embed, and store in the vector store.

        Args:
            embedding_fn: A callable that takes a list of texts and returns
                        a list of embedding vectors. If None, Chroma uses
                        its default embedding function.
            reset: If True, clear the collection before storing.

        Returns:
            The number of documents stored.

        === RAG Learning: Embedding Function Injection ===

        We accept embedding_fn as a parameter rather than hardcoding OpenAI.
        This pattern is called "dependency injection" and it's critical for:
          1. Testing — pass a mock that returns random vectors
          2. Flexibility — swap OpenAI for Cohere, local models, etc.
          3. Cost control — use cheaper embeddings for development
        """
        from src.core.learning_narrator import narrate

        if reset:
            self._store.reset()

        documents = self.prepare_documents()
        narrate("embedding.index", extra={"documents": len(documents)})

        if not documents:
            return 0

        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.doc_id for doc in documents]

        embeddings = None
        if embedding_fn is not None:
            embeddings = embedding_fn(texts)

        self._store.add_documents(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

        return len(documents)

    def get_document_stats(self) -> dict[str, Any]:
        """Return statistics about the prepared documents.

        Useful for verifying the embedding pipeline before running it.
        """
        documents = self.prepare_documents()
        doc_types: dict[str, int] = {}
        components_covered: set[str] = set()
        total_chars = 0

        for doc in documents:
            dtype = doc.metadata.get("doc_type", "unknown")
            doc_types[dtype] = doc_types.get(dtype, 0) + 1
            components_covered.add(doc.metadata.get("component_id", ""))
            total_chars += len(doc.text)

        return {
            "total_documents": len(documents),
            "doc_type_counts": doc_types,
            "components_covered": len(components_covered),
            "total_characters": total_chars,
            "avg_chars_per_doc": total_chars // len(documents) if documents else 0,
        }
