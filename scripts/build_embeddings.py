#!/usr/bin/env python3
"""One-time script to embed all component data into the Chroma vector store.

=== RAG Pipeline Learning: The Indexing Step ===

Every RAG system has two phases:
  1. INDEXING (offline): Convert your knowledge base into embeddings and store
     them in a vector database. This is what this script does.
  2. QUERYING (online): At query time, embed the user's question, search the
     vector store, and pass retrieved context to the LLM.

Indexing is typically done once, then incrementally updated as the knowledge
base changes. In our case, re-run this script whenever component YAML files
are added or modified.

Usage:
    # With OpenAI embeddings (requires OPENAI_API_KEY):
    python scripts/build_embeddings.py

    # With Chroma's default embeddings (no API key needed, good for dev):
    python scripts/build_embeddings.py --use-default-embeddings

    # Reset and rebuild from scratch:
    python scripts/build_embeddings.py --reset

    # Dry run — show stats without actually embedding:
    python scripts/build_embeddings.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.builder import DependencyGraph
from src.rag.vector_store import VectorStore
from src.rag.embedder import ComponentEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build component embeddings and store in Chroma vector database."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing embeddings and rebuild from scratch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show document stats without actually embedding.",
    )
    parser.add_argument(
        "--use-default-embeddings",
        action="store_true",
        help="Use Chroma's default embedding model instead of OpenAI.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model_config.yaml.",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    components_dir = PROJECT_ROOT / "data" / "components"
    variants_dir = PROJECT_ROOT / "data" / "variants"

    print("=" * 60)
    print("Impact Radar — Embedding Builder")
    print("=" * 60)

    # Step 1: Build dependency graph (to get component data)
    print("\n[1/4] Loading component data from YAML...")
    graph = DependencyGraph(components_dir, variants_dir)
    summary = graph.summary()
    print(f"  Loaded {summary['component_count']} components, "
          f"{summary['module_count']} modules, "
          f"{summary['variant_count']} variants")

    # Step 2: Initialize vector store
    print("\n[2/4] Initializing Chroma vector store...")
    store = VectorStore(config_path=config_path)
    print(f"  Collection: {store.collection_name}")
    print(f"  Existing documents: {store.count()}")

    # Step 3: Prepare documents
    print("\n[3/4] Preparing documents for embedding...")
    embedder = ComponentEmbedder(graph, store, config_path=config_path)
    stats = embedder.get_document_stats()
    print(f"  Total documents to embed: {stats['total_documents']}")
    print(f"  Document types:")
    for dtype, count in sorted(stats["doc_type_counts"].items()):
        print(f"    - {dtype}: {count}")
    print(f"  Components covered: {stats['components_covered']}")
    print(f"  Total characters: {stats['total_characters']:,}")
    print(f"  Avg chars/document: {stats['avg_chars_per_doc']}")

    if args.dry_run:
        print("\n[DRY RUN] Skipping embedding. Use without --dry-run to embed.")
        return

    # Step 4: Embed and store
    print("\n[4/4] Embedding and storing documents...")
    embedding_fn = None
    if not args.use_default_embeddings:
        try:
            from src.core.llm_client import LLMClient
            client = LLMClient(config_path=config_path)
            embedding_fn = client.get_embeddings_batch
            print(f"  Using OpenAI embeddings ({client.config.get('embedding', {}).get('model', 'unknown')})")
        except Exception as e:
            print(f"  WARNING: Could not initialize OpenAI client: {e}")
            print("  Falling back to Chroma default embeddings.")
            embedding_fn = None
    else:
        print("  Using Chroma default embeddings (sentence-transformers)")

    doc_count = embedder.embed_and_store(
        embedding_fn=embedding_fn,
        reset=args.reset,
    )

    print(f"\n  Stored {doc_count} documents in vector store")
    print(f"  Total documents in collection: {store.count()}")

    print("\n" + "=" * 60)
    print("Embedding complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
