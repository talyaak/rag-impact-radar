"""FastAPI REST layer for Impact Radar — exposes change-impact analysis via HTTP.

=== RAG Pipeline Learning: Serving a RAG System ===

A RAG pipeline has three phases: indexing, retrieval, and generation.
This API exposes all three:

  POST /api/v1/embeddings/rebuild  — triggers the INDEXING phase
       (re-embeds component data into the vector store)

  POST /api/v1/analyze             — triggers RETRIEVAL + GENERATION
       (graph traversal + semantic search + optional LLM synthesis)

  GET  /api/v1/components|variants|graph  — exposes the KNOWLEDGE GRAPH
       (the structured data that grounds everything)

The key design decision: initialize expensive resources (graph, vector store,
LLM client) ONCE at startup via the FastAPI lifespan, then share them across
requests. This avoids re-parsing YAML and re-connecting to Chroma on every
request — critical for latency in a production RAG serving layer.

=== Graceful Degradation ===

The /analyze endpoint implements a graceful degradation pattern common in
production RAG systems: if the LLM is unavailable (no API key, timeout,
quota exhausted), the endpoint still returns graph traversal + semantic
retrieval results with ``llm_used=False``. The structural and semantic
signals are valuable even without the LLM's natural-language synthesis.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph.builder import DependencyGraph
from src.graph.traverser import ImpactTraverser
from src.rag.vector_store import VectorStore
from src.rag.embedder import ComponentEmbedder
from src.rag.retriever import SemanticRetriever
from src.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "model_config.yaml"
_COMPONENTS_DIR = _PROJECT_ROOT / "data" / "components"
_VARIANTS_DIR = _PROJECT_ROOT / "data" / "variants"

# ── Shared state populated during lifespan ───────────────────────────────

_dep_graph: DependencyGraph | None = None
_vector_store: VectorStore | None = None
_retriever: SemanticRetriever | None = None
_llm_client: LLMClient | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ANN201, ARG001
    """Initialise heavy resources once at startup, tear down on shutdown.

    === RAG Pipeline Learning: Startup Initialisation ===

    Loading YAML files, building the NetworkX graph, and connecting to the
    Chroma vector store are all one-time costs.  By doing them inside the
    lifespan context manager we guarantee they happen exactly once and are
    available to every request handler via module-level references.
    """
    global _dep_graph, _vector_store, _retriever, _llm_client  # noqa: PLW0603

    logger.info("Building dependency graph from %s / %s", _COMPONENTS_DIR, _VARIANTS_DIR)
    _dep_graph = DependencyGraph(_COMPONENTS_DIR, _VARIANTS_DIR)

    logger.info("Connecting to vector store")
    _vector_store = VectorStore(config_path=_CONFIG_PATH)
    _retriever = SemanticRetriever.from_config(
        _vector_store,
        _vector_store.config,
    )

    try:
        _llm_client = LLMClient(config_path=_CONFIG_PATH)
        logger.info("LLM client initialised (model=%s)", _llm_client.model)
    except Exception:
        logger.warning("LLM client unavailable — analysis will run without LLM", exc_info=True)
        _llm_client = None

    yield  # ── application serves requests ──

    logger.info("Shutting down Impact Radar")


# ── FastAPI app ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Impact Radar",
    description=(
        "Change Impact Analyzer — surfaces non-obvious blast radius "
        "across product variants using graph traversal, semantic retrieval, "
        "and LLM-powered risk synthesis."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)

# ── Pydantic models ─────────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    """Request body for the ``/analyze`` endpoint."""

    changed_components: list[str] = Field(
        ...,
        description="Component IDs whose source has changed (e.g. ['auth_engine']).",
        min_length=1,
    )
    use_llm: bool = Field(
        True,
        description="Include LLM-generated risk narrative. Falls back to False on error.",
    )
    max_depth: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum graph-traversal depth (component-to-component hops).",
    )


class AnalyzeResponse(BaseModel):
    """Mirrors the dict returned by ``ImpactReporter.to_dict()``."""

    changed_components: list[str]
    direct_impacts: dict[str, Any]
    indirect_impacts: dict[str, Any]
    affected_variants: dict[str, Any]
    summary: dict[str, Any]
    semantic_matches: list[dict[str, Any]] = Field(default_factory=list)
    llm_narrative: str | None = None
    llm_used: bool = False


class EmbeddingsResponse(BaseModel):
    """Response from the ``/embeddings/rebuild`` endpoint."""

    documents_stored: int
    collection_name: str
    status: str


# ── Helpers ──────────────────────────────────────────────────────────────


def _require_graph() -> DependencyGraph:
    if _dep_graph is None:
        raise HTTPException(status_code=503, detail="Dependency graph not initialised.")
    return _dep_graph


def _require_vector_store() -> VectorStore:
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialised.")
    return _vector_store


def _require_retriever() -> SemanticRetriever:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised.")
    return _retriever


# ── Endpoints ────────────────────────────────────────────────────────────


@app.get("/health", description="Liveness / readiness probe.", tags=["ops"])
def health() -> dict[str, str]:
    """Return basic health status."""
    return {"status": "ok"}


@app.get(
    "/api/v1/components",
    description="List every registered component with its metadata.",
    tags=["graph"],
)
def list_components() -> list[dict[str, Any]]:
    """Return all components in the dependency graph.

    === RAG Pipeline Learning: Exposing the Knowledge Graph ===

    Making the raw graph data available via API lets callers inspect what
    the RAG system *knows* before asking it to analyse anything.  This
    transparency is a hallmark of trustworthy RAG systems — users can
    verify the facts the LLM will be grounded on.
    """
    graph = _require_graph()
    results: list[dict[str, Any]] = []
    for comp_id, comp in graph.get_components().items():
        results.append({
            "id": comp.id,
            "name": comp.name,
            "description": comp.description,
            "team_owner": comp.team_owner,
            "criticality": comp.criticality,
            "module_count": len(comp.modules),
        })
    return results


@app.get(
    "/api/v1/components/{component_id}",
    description="Retrieve a single component with its module dependencies.",
    tags=["graph"],
)
def get_component(component_id: str) -> dict[str, Any]:
    """Return one component including its full module list."""
    graph = _require_graph()
    comp = graph.get_component(component_id)
    if comp is None:
        raise HTTPException(status_code=404, detail=f"Component '{component_id}' not found.")
    modules = graph.get_component_modules(component_id)
    variants = graph.get_variants_for_component(component_id)
    return {
        "id": comp.id,
        "name": comp.name,
        "description": comp.description,
        "team_owner": comp.team_owner,
        "criticality": comp.criticality,
        "modules": modules,
        "api_surface": comp.api_surface,
        "variants": variants,
    }


@app.get(
    "/api/v1/variants",
    description="List every product variant.",
    tags=["graph"],
)
def list_variants() -> list[dict[str, Any]]:
    """Return all product variants."""
    graph = _require_graph()
    results: list[dict[str, Any]] = []
    for _vid, variant in graph.get_variants().items():
        results.append({
            "id": variant.id,
            "name": variant.name,
            "tier": variant.tier,
            "region": variant.region,
            "description": variant.description,
            "component_count": len(variant.components),
        })
    return results


@app.get(
    "/api/v1/variants/{variant_id}",
    description="Retrieve a single variant with its component list.",
    tags=["graph"],
)
def get_variant(variant_id: str) -> dict[str, Any]:
    """Return one variant including its full component manifest."""
    graph = _require_graph()
    variant = graph.get_variant(variant_id)
    if variant is None:
        raise HTTPException(status_code=404, detail=f"Variant '{variant_id}' not found.")
    return {
        "id": variant.id,
        "name": variant.name,
        "tier": variant.tier,
        "region": variant.region,
        "description": variant.description,
        "max_seats": variant.max_seats,
        "components": variant.components,
        "metadata": variant.metadata,
    }


@app.get(
    "/api/v1/graph/summary",
    description="High-level statistics about the dependency graph.",
    tags=["graph"],
)
def graph_summary() -> dict[str, Any]:
    """Return node/edge counts, shared modules, and variant totals.

    === RAG Pipeline Learning: Graph Introspection ===

    Before running an impact analysis it is useful to understand the shape
    of the knowledge graph — how many components, modules, edges, and
    shared modules exist.  This endpoint exposes ``DependencyGraph.summary()``
    which powers the sanity-check logging during startup.
    """
    graph = _require_graph()
    return graph.summary()


@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    description=(
        "Run a full change-impact analysis.  Combines graph traversal, "
        "semantic retrieval, and (optionally) LLM risk synthesis."
    ),
    tags=["analysis"],
)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Main analysis endpoint — the heart of Impact Radar.

    === RAG Pipeline Learning: Hybrid Retrieval in Action ===

    This endpoint orchestrates the complete RAG pipeline:
      1. GRAPH TRAVERSAL — ``ImpactTraverser`` walks the bipartite
         dependency graph to find structurally connected components.
      2. SEMANTIC RETRIEVAL — ``SemanticRetriever`` queries the vector
         store for conceptually related components the graph might miss.
      3. LLM GENERATION (optional) — ``LLMClient`` synthesises a
         human-readable risk narrative grounded in the retrieved evidence.

    If the LLM call fails the endpoint still returns the graph and semantic
    results with ``llm_used=False`` — graceful degradation.
    """
    graph = _require_graph()
    retriever = _require_retriever()

    # ── Validate requested components exist in the graph ─────────────
    known_ids = set(graph.get_components().keys())
    unknown = [c for c in request.changed_components if c not in known_ids]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown component(s): {', '.join(unknown)}. "
            f"Valid IDs: {', '.join(sorted(known_ids))}.",
        )

    # ── 1. Graph traversal ───────────────────────────────────────────
    traverser = ImpactTraverser(
        graph,
        max_depth=request.max_depth,
        include_weak_coupling=True,
    )
    impact_result = traverser.traverse(request.changed_components)

    # ── 2. Semantic retrieval ────────────────────────────────────────
    semantic_matches: list[dict[str, Any]] = []
    for comp_id in request.changed_components:
        comp = graph.get_component(comp_id)
        if comp is None:
            continue
        matches = retriever.find_related_to_component(
            component_id=comp_id,
            component_description=comp.description,
            exclude_component_ids=request.changed_components,
        )
        for match in matches:
            semantic_matches.append({
                "component_id": match.component_id,
                "component_name": match.component_name,
                "best_score": round(match.best_score, 4),
                "criticality": match.criticality,
                "match_count": match.match_count,
            })

    # ── 3. Optional LLM synthesis ────────────────────────────────────
    llm_narrative: str | None = None
    llm_used = False

    if request.use_llm and _llm_client is not None:
        try:
            prompt_context = _build_llm_prompt(
                request.changed_components, impact_result, semantic_matches
            )
            llm_narrative = _llm_client.generate(
                prompt=prompt_context,
                system_prompt=(
                    "You are a senior platform engineer assessing change-impact risk. "
                    "Explain the blast radius concisely, highlight non-obvious risks, "
                    "and suggest mitigation steps.  Ground every claim in the provided "
                    "evidence — do not speculate."
                ),
            )
            llm_used = True
        except Exception:
            logger.warning(
                "LLM generation failed — returning graph + semantic results only",
                exc_info=True,
            )

    # ── Build response ───────────────────────────────────────────────
    return AnalyzeResponse(
        changed_components=impact_result.changed_components,
        direct_impacts={
            cid: asdict(imp) for cid, imp in impact_result.direct_impacts.items()
        },
        indirect_impacts={
            cid: asdict(imp) for cid, imp in impact_result.indirect_impacts.items()
        },
        affected_variants={
            vid: asdict(vi) for vid, vi in impact_result.affected_variants.items()
        },
        summary=impact_result.summary(),
        semantic_matches=semantic_matches,
        llm_narrative=llm_narrative,
        llm_used=llm_used,
    )


@app.post(
    "/api/v1/embeddings/rebuild",
    response_model=EmbeddingsResponse,
    description="Re-embed all component data and rebuild the vector store index.",
    tags=["indexing"],
)
def rebuild_embeddings() -> EmbeddingsResponse:
    """Rebuild the vector store from current component YAML data.

    === RAG Pipeline Learning: Index Refresh ===

    Embeddings must stay in sync with the source data.  Whenever component
    YAML files change (new modules, updated descriptions) the vector store
    must be rebuilt so that semantic retrieval reflects the latest state.
    This endpoint makes that a single API call.
    """
    graph = _require_graph()
    store = _require_vector_store()

    embedder = ComponentEmbedder(graph, store, config_path=_CONFIG_PATH)
    doc_count = embedder.embed_and_store(reset=True)

    return EmbeddingsResponse(
        documents_stored=doc_count,
        collection_name=store.collection_name,
        status="ok",
    )


# ── Internal helpers ─────────────────────────────────────────────────────


def _build_llm_prompt(
    changed: list[str],
    impact_result: Any,
    semantic_matches: list[dict[str, Any]],
) -> str:
    """Assemble the retrieval-augmented prompt for the LLM.

    === RAG Pipeline Learning: Prompt Construction ===

    This is the "Augment" step of RAG.  We take the structured evidence
    produced by graph traversal and semantic retrieval and format it into
    a natural-language prompt the LLM can reason over.  Every fact in the
    prompt is traceable back to the knowledge graph or vector store —
    this is what prevents hallucination.
    """
    lines: list[str] = []
    lines.append(f"Changed components: {', '.join(changed)}")
    lines.append("")

    # Graph traversal evidence
    lines.append("## Direct impacts (shared module, 1 hop)")
    for cid, imp in impact_result.direct_impacts.items():
        paths_desc = "; ".join(p.describe() for p in imp.paths)
        lines.append(f"- {cid} (criticality={imp.criticality}): {paths_desc}")

    lines.append("")
    lines.append("## Indirect impacts (2+ hops)")
    for cid, imp in impact_result.indirect_impacts.items():
        paths_desc = "; ".join(p.describe() for p in imp.paths)
        lines.append(f"- {cid} (criticality={imp.criticality}): {paths_desc}")

    lines.append("")
    lines.append("## Affected product variants")
    for vid, vi in impact_result.affected_variants.items():
        lines.append(
            f"- {vi.variant_name} ({vid}): "
            f"{vi.direct_component_count} direct, "
            f"{vi.indirect_component_count} indirect, "
            f"max_criticality={vi.max_criticality}"
        )

    # Semantic retrieval evidence
    if semantic_matches:
        lines.append("")
        lines.append("## Semantically related components (vector similarity)")
        for sm in semantic_matches:
            lines.append(
                f"- {sm['component_name']} ({sm['component_id']}): "
                f"similarity={sm['best_score']}, criticality={sm['criticality']}"
            )

    lines.append("")
    lines.append(
        "Based on the evidence above, provide a concise risk assessment and "
        "recommended mitigation steps."
    )
    return "\n".join(lines)
