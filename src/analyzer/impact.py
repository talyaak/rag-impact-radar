"""Impact analysis orchestrator — wires graph traversal + semantic retrieval + LLM.

=== RAG Pipeline Learning: The Orchestration Layer ===

This is where the three pillars of impact-radar converge:

  1. GRAPH TRAVERSAL (structural retrieval)
     → "What components are CONNECTED to the changed ones?"
     → Uses BFS on the bipartite dependency graph.

  2. SEMANTIC RETRIEVAL (vector similarity)
     → "What components are CONCEPTUALLY related to the changed ones?"
     → Uses Chroma embeddings + cosine similarity.

  3. LLM GENERATION (grounded explanation)
     → "Explain the risk in plain English, using the evidence we found."
     → Sends structured prompt with both evidence types.

The orchestrator merges structural and semantic signals into a single scored
result set, then optionally asks the LLM to synthesize a risk narrative.
Components found by BOTH methods get a confidence boost — when two independent
retrieval strategies agree, the signal is stronger.

=== Why Scoring Matters ===

Raw traversal returns a flat list: "these components are affected." But a
1-hop tight coupling to a critical component is very different from a 5-hop
optional coupling to a low-priority one. The scoring formula turns qualitative
relationships into a quantitative severity number:

    score = coupling_weight × criticality_multiplier / depth

This lets the reporter sort impacts by urgency and lets the LLM focus its
explanation on the highest-risk paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.graph.builder import DependencyGraph
from src.graph.traverser import (
    ComponentImpact,
    ImpactPath,
    ImpactResult,
    ImpactTraverser,
)
from src.rag.retriever import ComponentMatch, SemanticRetriever
from src.core.llm_client import LLMClient
from src.core.privacy_guard import PrivacyGuard


# ── Data Classes ──────────────────────────────────────────────────────────


@dataclass
class ScoredImpact:
    """A single affected component with a computed severity score.

    === RAG Learning: Merging Two Retrieval Signals ===

    A ScoredImpact may carry evidence from the graph (paths, shared_modules)
    AND from semantic search (semantic_matches). When both signals are present,
    the component was found by two independent methods — this is high-confidence
    evidence. The score reflects both structural proximity and coupling strength;
    the semantic_matches add qualitative context for the LLM prompt.
    """

    component_id: str
    component_name: str
    criticality: str
    is_direct: bool
    depth: int
    score: float
    coupling: str  # strongest coupling in the path
    paths: list[ImpactPath]
    shared_modules: list[str]
    semantic_matches: list[ComponentMatch] = field(default_factory=list)


@dataclass
class VariantRisk:
    """Risk assessment for one product variant.

    === RAG Learning: Variant-Level Risk Aggregation ===

    Individual component impacts are useful, but stakeholders care about
    PRODUCT impact: "Is the Enterprise EU plan at risk?" This data class
    aggregates component-level scores into a variant-level risk picture,
    including the LLM's natural-language explanation of WHY the variant
    is at risk — the final output of the RAG pipeline.
    """

    variant_id: str
    variant_name: str
    tier: str
    region: str
    affected_components: list[ScoredImpact]
    direct_count: int
    indirect_count: int
    max_criticality: str
    risk_score: float
    llm_explanation: str = ""


@dataclass
class AnalysisResult:
    """Complete output of the impact analysis pipeline.

    === RAG Learning: The Pipeline Contract ===

    This is the single data structure that flows from the analyzer to the
    reporter. It contains everything needed to render any output format:
      - scored_impacts: for the component-level view
      - variant_risks: for the product-level view
      - graph_result: raw traversal data for debugging/auditing
      - semantic_matches: raw retrieval data for debugging/auditing
      - llm_used: whether the LLM was called (affects output formatting)

    Keeping raw data alongside scored data supports the "trust but verify"
    principle — users can inspect the raw evidence behind every score.
    """

    changed_components: list[str]
    scored_impacts: dict[str, ScoredImpact]
    variant_risks: dict[str, VariantRisk]
    graph_result: ImpactResult
    semantic_matches: list[ComponentMatch]
    llm_used: bool


# ── Coupling / criticality constants ──────────────────────────────────────

_CRITICALITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def _higher_criticality(a: str, b: str) -> str:
    return a if _CRITICALITY_RANK.get(a, 0) >= _CRITICALITY_RANK.get(b, 0) else b


# ── Analyzer ──────────────────────────────────────────────────────────────


class ImpactAnalyzer:
    """Orchestrates the full impact analysis: graph + RAG + LLM.

    === RAG Pipeline Learning: Orchestration Pattern ===

    The analyzer follows a strict pipeline:
      1. Structural retrieval (graph traversal)
      2. Semantic retrieval (vector search)
      3. Merge & score
      4. Variant risk aggregation
      5. LLM synthesis (optional)

    Each step is a pure function of its inputs — no global state, no side
    effects. This makes the pipeline testable (swap any component with a mock)
    and debuggable (inspect intermediate results at any stage).
    """

    def __init__(
        self,
        dep_graph: DependencyGraph,
        traverser: ImpactTraverser,
        retriever: SemanticRetriever | None = None,
        llm_client: LLMClient | None = None,
        config_path: str | Path = "config/model_config.yaml",
        privacy_guard: PrivacyGuard | None = None,
    ) -> None:
        self._graph = dep_graph
        self._traverser = traverser
        self._retriever = retriever
        self._llm = llm_client
        self._privacy_guard = privacy_guard
        self._config = self._load_config(config_path)

        # Scoring weights from config
        analysis = self._config.get("analysis", {})
        weights = analysis.get("severity_weights", {})
        self._coupling_weights = {
            "tight": weights.get("coupling_tight", 1.0),
            "loose": weights.get("coupling_loose", 0.5),
            "optional": weights.get("coupling_optional", 0.2),
        }
        mults = analysis.get("criticality_multipliers", {})
        self._criticality_multipliers = {
            "critical": mults.get("critical", 4.0),
            "high": mults.get("high", 2.0),
            "medium": mults.get("medium", 1.0),
            "low": mults.get("low", 0.5),
        }

    @staticmethod
    def _load_config(config_path: str | Path) -> dict[str, Any]:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def analyze(
        self,
        changed_components: list[str],
        use_llm: bool = True,
    ) -> AnalysisResult:
        """Run the complete impact analysis pipeline.

        Args:
            changed_components: Component IDs that changed.
            use_llm: If True and LLM client available, generate risk narratives.

        Returns:
            AnalysisResult with scored impacts, variant risks, and optional
            LLM explanations.

        === RAG Learning: The Pipeline in Action ===

        This method is the beating heart of the RAG system. Watch the data
        transform through each stage:

          Input:  ["auth_engine", "billing_core"]
            ↓
          Graph:  ImpactResult with 10 affected components, paths, variants
            ↓
          RAG:    5 semantically similar components (some overlap with graph)
            ↓
          Score:  Each impact gets a severity number (0-16+)
            ↓
          Merge:  Graph-only + RAG-only + both-confirmed impacts
            ↓
          LLM:    "Enterprise EU is at risk because..." (grounded narrative)
            ↓
          Output: AnalysisResult ready for the reporter
        """
        from src.core.learning_narrator import narrate

        narrate("analyze.retrieval", extra={"changed": ",".join(changed_components)})

        # ── 1. Graph traversal ──────────────────────────────────────
        graph_result = self._traverser.traverse(changed_components)

        # ── 2. Semantic retrieval ───────────────────────────────────
        all_semantic: list[ComponentMatch] = []
        if self._retriever is not None:
            for comp_id in changed_components:
                comp = self._graph.get_component(comp_id)
                if comp is None:
                    continue
                matches = self._retriever.find_related_to_component(
                    component_id=comp_id,
                    component_description=comp.description,
                    exclude_component_ids=changed_components,
                )
                all_semantic.extend(matches)

        # Deduplicate semantic matches by component_id (keep best score)
        semantic_by_comp: dict[str, ComponentMatch] = {}
        for match in all_semantic:
            existing = semantic_by_comp.get(match.component_id)
            if existing is None or match.best_score > existing.best_score:
                semantic_by_comp[match.component_id] = match

        # ── 3. Score and merge ──────────────────────────────────────
        scored_impacts: dict[str, ScoredImpact] = {}

        # Score graph-discovered impacts
        all_graph_impacts = {
            **graph_result.direct_impacts,
            **graph_result.indirect_impacts,
        }
        for comp_id, impact in all_graph_impacts.items():
            comp = self._graph.get_component(comp_id)
            comp_name = comp.name if comp else comp_id
            score = self._score_impact(impact)

            # Check if this component was also found by semantic search
            sem_matches = []
            if comp_id in semantic_by_comp:
                sem_matches = [semantic_by_comp.pop(comp_id)]
                # Boost score for components found by both methods
                score *= 1.25

            # Determine strongest coupling in paths
            coupling = "optional"
            for path in impact.paths:
                if path.min_coupling == "tight":
                    coupling = "tight"
                    break
                if path.min_coupling == "loose" and coupling != "tight":
                    coupling = "loose"

            scored_impacts[comp_id] = ScoredImpact(
                component_id=comp_id,
                component_name=comp_name,
                criticality=impact.criticality,
                is_direct=impact.is_direct,
                depth=impact.depth,
                score=round(score, 2),
                coupling=coupling,
                paths=impact.paths,
                shared_modules=impact.shared_modules,
                semantic_matches=sem_matches,
            )

        # Add semantic-only discoveries (not found by graph)
        for comp_id, match in semantic_by_comp.items():
            comp = self._graph.get_component(comp_id)
            criticality = comp.criticality if comp else "medium"
            crit_mult = self._criticality_multipliers.get(criticality, 1.0)
            score = match.best_score * crit_mult

            scored_impacts[comp_id] = ScoredImpact(
                component_id=comp_id,
                component_name=match.component_name,
                criticality=criticality,
                is_direct=False,
                depth=0,  # No graph path — semantic only
                score=round(score, 2),
                coupling="semantic",
                paths=[],
                shared_modules=[],
                semantic_matches=[match],
            )

        # ── 4. Variant risk aggregation ─────────────────────────────
        variant_risks = self._build_variant_risks(
            changed_components, scored_impacts, graph_result
        )

        # ── 5. Optional LLM synthesis ───────────────────────────────
        llm_used = False
        if use_llm and self._llm is not None:
            try:
                self._generate_explanations(
                    changed_components,
                    graph_result,
                    list(semantic_by_comp.values()) + all_semantic,
                    variant_risks,
                )
                llm_used = True
            except Exception:
                pass  # Graceful degradation — results still valid without LLM

        return AnalysisResult(
            changed_components=changed_components,
            scored_impacts=scored_impacts,
            variant_risks=variant_risks,
            graph_result=graph_result,
            semantic_matches=all_semantic,
            llm_used=llm_used,
        )

    def _score_impact(self, impact: ComponentImpact) -> float:
        """Calculate severity score for a graph-discovered impact.

        Formula: coupling_weight × criticality_multiplier / depth

        === RAG Learning: Why This Formula? ===

        - coupling_weight: tight (1.0) > loose (0.5) > optional (0.2)
          A tight dependency means the impacted component WILL break.
          An optional one might not even be active.

        - criticality_multiplier: critical (4.0) > high (2.0) > medium (1.0) > low (0.5)
          Breaking the billing system is worse than breaking a debug tool.

        - / depth: closer impacts are more severe. A direct dependency
          (depth=1) gets full score; a 3-hop indirect gets 1/3.

        Example: tight coupling to critical component at depth 1:
          1.0 × 4.0 / 1 = 4.0
        vs optional coupling to low component at depth 3:
          0.2 × 0.5 / 3 = 0.03
        """
        # Use strongest coupling from all paths
        best_coupling = "optional"
        for path in impact.paths:
            if path.min_coupling == "tight":
                best_coupling = "tight"
                break
            if path.min_coupling == "loose" and best_coupling != "tight":
                best_coupling = "loose"

        coupling_w = self._coupling_weights.get(best_coupling, 0.2)
        crit_mult = self._criticality_multipliers.get(impact.criticality, 1.0)
        depth = max(impact.depth, 1)  # Avoid division by zero

        return coupling_w * crit_mult / depth

    def _build_variant_risks(
        self,
        changed_components: list[str],
        scored_impacts: dict[str, ScoredImpact],
        graph_result: ImpactResult,
    ) -> dict[str, VariantRisk]:
        """Aggregate component scores into variant-level risk.

        A variant's risk_score is the sum of its affected component scores.
        This means a variant with many low-severity impacts can rank higher
        than one with a single high-severity impact — reflecting the
        cumulative risk of widespread changes.
        """
        variant_risks: dict[str, VariantRisk] = {}

        for vid, vi in graph_result.affected_variants.items():
            variant = self._graph.get_variant(vid)
            if variant is None:
                continue

            affected: list[ScoredImpact] = []
            risk_score = 0.0
            direct_count = 0
            indirect_count = 0
            max_crit = "low"

            for comp_id in vi.affected_components:
                if comp_id in scored_impacts:
                    si = scored_impacts[comp_id]
                    affected.append(si)
                    risk_score += si.score
                    if si.is_direct or comp_id in changed_components:
                        direct_count += 1
                    else:
                        indirect_count += 1
                    max_crit = _higher_criticality(max_crit, si.criticality)
                elif comp_id in changed_components:
                    # Changed components aren't in scored_impacts
                    direct_count += 1
                    comp = self._graph.get_component(comp_id)
                    if comp:
                        max_crit = _higher_criticality(max_crit, comp.criticality)

            variant_risks[vid] = VariantRisk(
                variant_id=vid,
                variant_name=variant.name,
                tier=variant.tier,
                region=variant.region,
                affected_components=affected,
                direct_count=direct_count,
                indirect_count=indirect_count,
                max_criticality=max_crit,
                risk_score=round(risk_score, 2),
            )

        return variant_risks

    def _generate_explanations(
        self,
        changed_components: list[str],
        graph_result: ImpactResult,
        semantic_matches: list[ComponentMatch],
        variant_risks: dict[str, VariantRisk],
    ) -> None:
        """Generate LLM risk explanations for each affected variant.

        Mutates variant_risks in place by setting llm_explanation.

        === RAG Learning: One Prompt Per Variant ===

        We generate one explanation per variant rather than one giant
        explanation for everything. This keeps each prompt focused and
        within context window limits. Each prompt includes only the
        evidence relevant to THAT variant's affected components.
        """
        from src.core.learning_narrator import narrate

        assert self._llm is not None

        narrate("analyze.generation", extra={"variants": len(variant_risks)})
        system_prompt = self._build_system_prompt()

        for vid, vr in variant_risks.items():
            if not vr.affected_components:
                continue

            prompt = self._build_llm_prompt(
                changed_components, graph_result, semantic_matches, vr
            )
            try:
                if self._privacy_guard is not None:
                    explanation = self._privacy_guard.guarded_generate(
                        self._llm,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=0.2,
                        max_tokens=512,
                    )
                else:
                    explanation = self._llm.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=0.2,
                        max_tokens=512,
                    )
                vr.llm_explanation = explanation.strip()
            except Exception:
                vr.llm_explanation = ""

    def _build_system_prompt(self) -> str:
        """System prompt that sets the LLM's role and constraints.

        === RAG Learning: Constraining the LLM ===

        The system prompt does three things:
          1. Sets the role (risk analyst, not creative writer)
          2. Sets the output format (concise, structured)
          3. Sets the grounding constraint (cite evidence, don't speculate)

        Without these constraints, the LLM might write a creative essay
        about software risk. With them, it produces actionable analysis
        tied to specific dependency paths.
        """
        return (
            "You are a senior platform risk analyst for NexusSaaS. "
            "Your job is to assess the impact of component changes on product variants. "
            "Rules:\n"
            "- Ground every claim in the evidence provided. Do NOT speculate.\n"
            "- Use uncertainty markers: 'confirmed' for direct dependencies, "
            "'likely' for indirect, 'possible' for semantic-only matches.\n"
            "- Keep explanations concise — 2-4 sentences per variant.\n"
            "- Highlight non-obvious risks that a developer might miss.\n"
            "- Suggest specific mitigation steps when possible."
        )

    def _build_llm_prompt(
        self,
        changed_components: list[str],
        graph_result: ImpactResult,
        semantic_matches: list[ComponentMatch],
        variant_risk: VariantRisk,
    ) -> str:
        """Build a structured prompt with evidence for one variant.

        === RAG Learning: The Augmented Prompt ===

        This is the "A" in RAG — Augmentation. We take raw evidence from
        two retrieval methods and format it into labeled sections that the
        LLM can reference. The structure is deliberate:

          ## Change Scope       ← What changed (input)
          ## Structural Evidence ← What the graph found (retrieval 1)
          ## Semantic Evidence   ← What vector search found (retrieval 2)
          ## Task                ← What to generate (instruction)

        Each section is clearly labeled so the LLM knows which evidence
        type it's looking at. This prevents the LLM from confusing graph
        paths (structural facts) with semantic matches (similarity signals).
        """
        lines: list[str] = []

        # Change scope
        lines.append("## Change Scope")
        lines.append(f"Components changed: {', '.join(changed_components)}")
        lines.append(f"Variant under assessment: {variant_risk.variant_name} "
                      f"({variant_risk.tier}, {variant_risk.region})")
        lines.append("")

        # Structural evidence from graph
        lines.append("## Structural Evidence (dependency graph)")
        affected_ids = {si.component_id for si in variant_risk.affected_components}
        for si in variant_risk.affected_components:
            if si.paths:
                for path in si.paths[:2]:  # Limit to 2 paths per component
                    lines.append(f"- {path.describe()}")
            if si.shared_modules:
                mods = ", ".join(si.shared_modules)
                lines.append(f"  Shared modules: {mods} (coupling: {si.coupling})")

        # Add any relevant paths from graph_result not in variant
        for path in graph_result.all_impact_paths:
            if path.target in affected_ids and len(lines) < 30:
                lines.append(f"- {path.describe()}")
        lines.append("")

        # Semantic evidence
        relevant_semantic = [
            m for m in semantic_matches
            if m.component_id in affected_ids
        ]
        if relevant_semantic:
            lines.append("## Semantic Evidence (vector similarity)")
            for match in relevant_semantic[:3]:
                evidence = ""
                if match.matching_docs:
                    evidence = f" — \"{match.matching_docs[0].document[:120]}\""
                lines.append(
                    f"- {match.component_name} (similarity={match.best_score:.2f})"
                    f"{evidence}"
                )
            lines.append("")

        # Task
        lines.append("## Task")
        lines.append(
            f"Explain why the {variant_risk.variant_name} variant is at risk "
            f"from the changes to {', '.join(changed_components)}. "
            "Highlight any non-obvious indirect dependencies. "
            "Suggest mitigation steps."
        )

        return "\n".join(lines)
