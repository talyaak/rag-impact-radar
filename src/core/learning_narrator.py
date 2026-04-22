"""Learning-mode narrator — prints short "why we do this" explainers.

Silent by default. When ``LEARNING_MODE=1`` is set in the environment
(either via shell, ``.env``, or an explicit ``enabled=True`` to the
programmatic API), each narrated lifecycle event prints a small fenced
block explaining what just happened and pointing at the relevant
``docs/0X-*.md`` chapter.

The corpus is copied and condensed from the existing
``=== RAG Pipeline Learning: ...`` docstrings scattered across
``src/rag/embedder.py``, ``src/rag/retriever.py``, ``src/graph/builder.py``,
``src/core/llm_client.py``, ``src/api/main.py``, and
``scripts/build_embeddings.py`` — no new prose, just consolidation.

Contract:

    from src.core.learning_narrator import narrate
    narrate("ingest.scan")   # prints if LEARNING_MODE=1, else no-op.

Unknown topics are a safe no-op (not a crash), so threading new
``narrate()`` calls through untested code paths can't break anything
if a topic slug is missing from the dictionary.
"""

from __future__ import annotations

import os
import sys
from typing import Any, TextIO

_NARRATIONS: dict[str, dict[str, str]] = {
    "startup": {
        "summary": (
            "Impact Radar is booting up. Indexing, retrieval, and generation — "
            "the three phases of a RAG system — all share expensive resources "
            "(the dependency graph, the vector store, the LLM client). We "
            "initialise them once in the FastAPI lifespan so every request "
            "reuses the same in-memory graph and connection pool."
        ),
        "chapter": "docs/05-retrieval.md",
    },
    "seed_data_loaded": {
        "summary": (
            "YAML under data/components/ and data/variants/ has been parsed "
            "into a NetworkX bipartite graph. Components are the nodes that "
            "publish APIs; variants are the product-specific overlays; modules "
            "are the shared libraries that wire them together. The graph is "
            "the structural ground truth that anchors every retrieval result."
        ),
        "chapter": "docs/03-graph.md",
    },
    "ingest.scan": {
        "summary": (
            "Scanning the repository tree. The scanner classifies files by "
            "extension and name — Python sources, manifests (package.json, "
            ".csproj, .sln), docs, config — and records service boundaries "
            "(directories marked by Dockerfiles, manifests, build files). "
            "No parsing yet; this is just discovery and classification."
        ),
        "chapter": "docs/07-v2-evolution.md",
    },
    "ingest.parse": {
        "summary": (
            "Parsing discovered files into components and modules. Python "
            "uses the stdlib AST module to extract classes that meet the "
            "min-methods heuristic. TypeScript/JavaScript and C# are "
            "manifest-driven: one component per package.json or .csproj, "
            "because built artifacts — not source classes — are the unit "
            "teams actually reason about in polyglot shops."
        ),
        "chapter": "docs/07-v2-evolution.md",
    },
    "ingest.graph": {
        "summary": (
            "Merging newly parsed components into the dependency graph. "
            "Existing YAML under data/components/ loads first, then parsed "
            "components fill in. Gaps between what was asserted in YAML and "
            "what the code actually imports surface as high-signal prompts "
            "during the gap-analysis phase."
        ),
        "chapter": "docs/03-graph.md",
    },
    "embedding.index": {
        "summary": (
            "Embedding component descriptions into the vector store. Each "
            "component body gets chunked (overlap preserves boundary context), "
            "embedded once per content-hash (cached on disk to avoid "
            "re-billing OpenAI for identical text), then written to Chroma. "
            "This is the INDEXING phase of the RAG lifecycle."
        ),
        "chapter": "docs/04-embeddings.md",
    },
    "gaps.detect": {
        "summary": (
            "Running graph-based heuristics to find missing metadata: "
            "components without team_owner, orphaned modules, untyped "
            "couplings, under-described APIs. Detection is deterministic "
            "— no LLM involved — because graph completeness is structural, "
            "not subjective."
        ),
        "chapter": "docs/07-v2-evolution.md",
    },
    "gaps.suggest": {
        "summary": (
            "Asking the LLM to propose answers for the detected gaps. "
            "Suggestions are batched (many gaps per call) rather than one "
            "call per gap — that's typically a ~10x cost reduction during "
            "onboarding. The user accepts or overrides; nothing is written "
            "to YAML without explicit confirmation."
        ),
        "chapter": "docs/07-v2-evolution.md",
    },
    "compile.export": {
        "summary": (
            "Writing the merged graph (gaps filled, suggestions accepted) "
            "back to data/components/*.yaml and data/variants/*.yaml. This "
            "is the 'recompile' step: what started as scanned code plus "
            "seed data becomes a durable, version-controllable snapshot of "
            "the system's structural knowledge."
        ),
        "chapter": "docs/07-v2-evolution.md",
    },
    "analyze.retrieval": {
        "summary": (
            "First RAG phase: RETRIEVAL. The user's change description is "
            "embedded, the vector store returns the top-k most relevant "
            "components, and graph traversal expands outward along "
            "coupling edges to find downstream blast radius. Two signals — "
            "semantic similarity and structural reachability — combine "
            "into the candidate set."
        ),
        "chapter": "docs/05-retrieval.md",
    },
    "analyze.generation": {
        "summary": (
            "Second RAG phase: GENERATION. The LLM is handed the retrieved "
            "context and asked to synthesise a risk explanation grounded "
            "in that evidence. If the LLM is unavailable, the endpoint "
            "still returns the structural+semantic signals with "
            "llm_used=False — graceful degradation is a first-class "
            "feature of production RAG systems."
        ),
        "chapter": "docs/06-llm-generation.md",
    },
}


def is_enabled() -> bool:
    """True when LEARNING_MODE is set to a truthy value in the environment."""
    raw = os.environ.get("LEARNING_MODE", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def narrate(
    topic: str,
    *,
    enabled: bool | None = None,
    stream: TextIO | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Print a narration block for ``topic`` when learning mode is on.

    Args:
        topic: One of the keys in ``_NARRATIONS`` (e.g. ``"ingest.scan"``).
            Unknown topics are a silent no-op.
        enabled: Override the ``LEARNING_MODE`` env var check. ``None``
            (default) defers to the environment.
        stream: Output stream. Defaults to ``sys.stdout``.
        extra: Optional dict appended as a ``"Context:"`` line (e.g. a
            path or count the caller wants visible in the block).
    """
    if enabled is None:
        enabled = is_enabled()
    if not enabled:
        return
    entry = _NARRATIONS.get(topic)
    if entry is None:
        return
    out = stream or sys.stdout
    print(f"\n┌─ RAG Pipeline Learning: {topic} ─", file=out)
    print(f"│ {entry['summary']}", file=out)
    print(f"│ Why we do this: see {entry['chapter']}.", file=out)
    if extra:
        for key, value in extra.items():
            print(f"│ {key}: {value}", file=out)
    print("└" + "─" * 60, file=out)
