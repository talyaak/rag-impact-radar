"""Impact traverser — finds all affected components and variants given a change scope.

=== RAG Pipeline Learning: Graph Traversal as "Structured Retrieval" ===

In a classic RAG pipeline, retrieval means "find relevant documents via
vector similarity." But vector similarity is SEMANTIC — it finds things
that SOUND related. Graph traversal is STRUCTURAL — it finds things that
ARE related through explicit dependency chains.

Consider this example:
  - Vector search for "AuthEngine" might return "EncryptionService" (both
    deal with security) — that's a useful semantic match.
  - Graph traversal from "AuthEngine" finds "WebSocketGateway" (they share
    session_store) — that's a structural fact vector search would miss.

The magic of impact-radar is combining both: graph traversal catches
structural blast radius, semantic search catches conceptual blast radius,
and the LLM synthesizes both into a human-readable risk report.

=== Why BFS (Breadth-First Search)? ===

We use BFS rather than DFS because:
  1. BFS finds the SHORTEST path first — important for ranking severity
     (a 1-hop dependency is higher risk than a 5-hop dependency)
  2. BFS naturally produces "depth levels" — we can classify impacts as
     direct (depth 1) vs. indirect (depth 2+)
  3. BFS with a depth limit is straightforward — just stop expanding
     beyond max_depth

The traversal alternates between node types in the bipartite graph:
  component → module → component → module → ...
Each "hop" crosses one edge, so depth 1 = same module, depth 2 = one
intermediate module apart, etc.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.graph.builder import DependencyGraph


@dataclass
class ImpactPath:
    """A single path showing how impact propagates from source to target.

    The chain alternates: [component, module, component, module, component]
    Each element is a node ID in the bipartite graph.

    === RAG Learning: Paths as LLM Context ===

    These paths become part of the prompt sent to the LLM in the analysis
    phase. Instead of asking "is AuthEngine related to WebSocketGateway?"
    (which might hallucinate), we provide the path:
      AuthEngine → session_store → WebSocketGateway
    and ask the LLM to EXPLAIN why this path matters — grounded generation.
    """

    source: str
    target: str
    chain: list[str]
    depth: int  # Number of component-to-component hops (not raw edges)
    min_coupling: str  # Weakest coupling in the chain (tight > loose > optional)

    def describe(self) -> str:
        """Human-readable description of the impact path."""
        arrows = " → ".join(self.chain)
        return f"{arrows} (depth={self.depth}, min_coupling={self.min_coupling})"


@dataclass
class ComponentImpact:
    """Impact assessment for a single affected component."""

    component_id: str
    is_direct: bool  # True if shares a module with a changed component
    depth: int  # Shortest path depth from any changed component
    paths: list[ImpactPath]  # All paths from changed components to this one
    shared_modules: list[str]  # Modules shared with changed components
    criticality: str  # Component's criticality level


@dataclass
class VariantImpact:
    """Impact assessment for a single affected variant."""

    variant_id: str
    variant_name: str
    affected_components: list[str]  # Component IDs in this variant that are affected
    direct_component_count: int
    indirect_component_count: int
    max_criticality: str  # Highest criticality among affected components


@dataclass
class ImpactResult:
    """Complete impact analysis result.

    This is the structured output from graph traversal that gets passed to:
    1. The RAG retriever (to enrich with semantic context)
    2. The LLM (to generate risk explanations)
    3. The reporter (to format human-readable output)
    """

    changed_components: list[str]
    direct_impacts: dict[str, ComponentImpact]
    indirect_impacts: dict[str, ComponentImpact]
    affected_variants: dict[str, VariantImpact]
    all_impact_paths: list[ImpactPath]

    @property
    def all_affected_components(self) -> list[str]:
        """All affected component IDs (direct + indirect), not including changed ones."""
        return sorted(
            set(list(self.direct_impacts.keys()) + list(self.indirect_impacts.keys()))
        )

    @property
    def total_affected_variants(self) -> int:
        return len(self.affected_variants)

    def summary(self) -> dict[str, Any]:
        """Quick summary for logging/debugging."""
        return {
            "changed": self.changed_components,
            "direct_impacts": len(self.direct_impacts),
            "indirect_impacts": len(self.indirect_impacts),
            "total_affected_components": len(self.all_affected_components),
            "affected_variants": len(self.affected_variants),
            "total_paths": len(self.all_impact_paths),
        }


# Coupling strength ordering for min-coupling calculation
_COUPLING_RANK = {"tight": 3, "loose": 2, "optional": 1}


def _weaker_coupling(a: str, b: str) -> str:
    """Return the weaker of two coupling strengths."""
    return a if _COUPLING_RANK.get(a, 0) <= _COUPLING_RANK.get(b, 0) else b


def _strongest_coupling(couplings: list[str]) -> str:
    """Return the strongest coupling from a list."""
    if not couplings:
        return "optional"
    best = couplings[0]
    for c in couplings[1:]:
        if _COUPLING_RANK.get(c, 0) > _COUPLING_RANK.get(best, 0):
            best = c
    return best


_CRITICALITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def _higher_criticality(a: str, b: str) -> str:
    """Return the higher of two criticality levels."""
    return a if _CRITICALITY_RANK.get(a, 0) >= _CRITICALITY_RANK.get(b, 0) else b


class ImpactTraverser:
    """Traverses the dependency graph to find all components and variants
    affected by a set of changed components.

    === RAG Learning: Traversal Produces "Retrieval Evidence" ===

    In a RAG pipeline, you need EVIDENCE to ground the LLM's generation.
    The traverser produces that evidence:
      - Impact paths show exactly HOW component A relates to component B
      - Coupling metadata shows HOW STRONGLY they're related
      - Depth shows HOW FAR the impact travels

    Without this evidence, asking an LLM "what's the blast radius of changing
    AuthEngine?" would produce plausible-sounding but potentially hallucinated
    answers. With it, the LLM explains known structural facts.
    """

    def __init__(
        self,
        dep_graph: DependencyGraph,
        max_depth: int = 5,
        include_weak_coupling: bool = True,
    ) -> None:
        self._dep_graph = dep_graph
        self._max_depth = max_depth
        self._include_weak_coupling = include_weak_coupling

    def traverse(
        self,
        changed_components: list[str],
        max_depth: int | None = None,
    ) -> ImpactResult:
        """Find all components and variants affected by the given changes.

        Uses BFS on the bipartite graph, alternating between component and
        module nodes. Each full "hop" = component → module → component.

        Args:
            changed_components: List of component IDs that changed.
            max_depth: Override max traversal depth (in component-to-component hops).

        Returns:
            ImpactResult with direct impacts, indirect impacts, paths, and variants.
        """
        effective_depth = max_depth if max_depth is not None else self._max_depth
        graph = self._dep_graph.graph

        # Validate inputs
        valid_changed = [c for c in changed_components if c in graph]

        # BFS state
        # We track visited COMPONENT nodes to prevent cycles.
        # Module nodes are intermediate — we pass through them but don't "visit" them.
        visited_components: set[str] = set(valid_changed)
        all_paths: list[ImpactPath] = []

        # impacts[component_id] = ComponentImpact
        impacts: dict[str, ComponentImpact] = {}

        # BFS queue entries: (current_component, depth, path_so_far)
        # path_so_far is a list of node IDs alternating component/module
        queue: deque[tuple[str, int, list[str]]] = deque()

        # Seed the BFS with all changed components at depth 0
        for comp_id in valid_changed:
            queue.append((comp_id, 0, [comp_id]))

        while queue:
            current_comp, depth, path = queue.popleft()

            # Don't expand beyond max depth
            if depth >= effective_depth:
                continue

            # Find all modules this component connects to
            for module_id in graph.neighbors(current_comp):
                node_data = graph.nodes[module_id]
                if node_data.get("node_type") != "module":
                    continue

                # Check coupling filter
                edge_data = graph.edges[current_comp, module_id]
                coupling = edge_data.get("coupling", "tight")
                if not self._include_weak_coupling and coupling in (
                    "loose",
                    "optional",
                ):
                    continue

                # Find all OTHER components connected to this module
                for neighbor_comp in graph.neighbors(module_id):
                    neighbor_data = graph.nodes[neighbor_comp]
                    if neighbor_data.get("node_type") != "component":
                        continue

                    # Skip if it's a changed component (source) or already visited
                    if neighbor_comp in visited_components:
                        continue

                    # Found a new affected component!
                    visited_components.add(neighbor_comp)
                    new_depth = depth + 1
                    new_path = path + [module_id, neighbor_comp]

                    # Calculate min coupling along this path
                    min_coupling = self._path_min_coupling(new_path)

                    impact_path = ImpactPath(
                        source=valid_changed[0] if len(valid_changed) == 1 else path[0],
                        target=neighbor_comp,
                        chain=list(new_path),
                        depth=new_depth,
                        min_coupling=min_coupling,
                    )
                    all_paths.append(impact_path)

                    # Determine which modules are shared with changed components
                    shared_mods = self._find_shared_modules(
                        neighbor_comp, valid_changed
                    )

                    criticality = graph.nodes[neighbor_comp].get(
                        "criticality", "medium"
                    )

                    impact = ComponentImpact(
                        component_id=neighbor_comp,
                        is_direct=(new_depth == 1),
                        depth=new_depth,
                        paths=[impact_path],
                        shared_modules=shared_mods,
                        criticality=criticality,
                    )
                    impacts[neighbor_comp] = impact

                    # Continue BFS from this newly discovered component
                    queue.append((neighbor_comp, new_depth, new_path))

        # Split into direct (depth 1) and indirect (depth 2+)
        direct = {
            cid: imp for cid, imp in impacts.items() if imp.is_direct
        }
        indirect = {
            cid: imp for cid, imp in impacts.items() if not imp.is_direct
        }

        # Map affected components to affected variants
        affected_variants = self._compute_affected_variants(
            valid_changed, impacts
        )

        return ImpactResult(
            changed_components=valid_changed,
            direct_impacts=direct,
            indirect_impacts=indirect,
            affected_variants=affected_variants,
            all_impact_paths=all_paths,
        )

    def _path_min_coupling(self, path: list[str]) -> str:
        """Find the weakest coupling along a path through the graph.

        The path alternates: [component, module, component, module, ...]
        Edges exist between adjacent component-module pairs.
        """
        graph = self._dep_graph.graph
        min_coup = "tight"
        for i in range(len(path) - 1):
            if graph.has_edge(path[i], path[i + 1]):
                edge_coupling = graph.edges[path[i], path[i + 1]].get(
                    "coupling", "tight"
                )
                min_coup = _weaker_coupling(min_coup, edge_coupling)
        return min_coup

    def _find_shared_modules(
        self, component_id: str, changed_components: list[str]
    ) -> list[str]:
        """Find modules that a component shares with any changed component."""
        graph = self._dep_graph.graph
        comp_modules = {
            n
            for n in graph.neighbors(component_id)
            if graph.nodes[n].get("node_type") == "module"
        }
        changed_modules: set[str] = set()
        for changed in changed_components:
            for n in graph.neighbors(changed):
                if graph.nodes[n].get("node_type") == "module":
                    changed_modules.add(n)
        return sorted(comp_modules & changed_modules)

    def _compute_affected_variants(
        self,
        changed_components: list[str],
        impacts: dict[str, ComponentImpact],
    ) -> dict[str, VariantImpact]:
        """Determine which product variants are affected and how.

        A variant is affected if it contains ANY changed or impacted component.
        """
        # Collect all affected component IDs (changed + impacted)
        all_affected = set(changed_components) | set(impacts.keys())

        variant_impacts: dict[str, VariantImpact] = {}

        for variant_id, variant in self._dep_graph.get_variants().items():
            variant_comp_ids = {c["component_id"] for c in variant.components}
            affected_in_variant = variant_comp_ids & all_affected

            if not affected_in_variant:
                continue

            # Count direct vs indirect
            direct_count = 0
            indirect_count = 0
            max_crit = "low"

            for comp_id in affected_in_variant:
                if comp_id in changed_components:
                    direct_count += 1
                elif comp_id in impacts and impacts[comp_id].is_direct:
                    direct_count += 1
                else:
                    indirect_count += 1

                # Track max criticality
                comp = self._dep_graph.get_component(comp_id)
                if comp:
                    max_crit = _higher_criticality(max_crit, comp.criticality)

            variant_impacts[variant_id] = VariantImpact(
                variant_id=variant_id,
                variant_name=variant.name,
                affected_components=sorted(affected_in_variant),
                direct_component_count=direct_count,
                indirect_component_count=indirect_count,
                max_criticality=max_crit,
            )

        return variant_impacts
