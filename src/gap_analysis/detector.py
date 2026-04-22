"""Gap detector — identifies structural gaps, orphans, and ambiguities in the ingested graph.

After the ingestion engine builds an initial graph from code analysis, the gap
detector examines it for:
  - Orphan components (no module connections)
  - Isolated modules (used by only one component)
  - Missing descriptions (empty or auto-generated)
  - Coupling ambiguity (all edges default to "tight")
  - Weak coverage (components with no API surface)
  - Missing variants (components not assigned to any variant)
  - Potential missing links (components in same directory but not connected)

Each gap becomes a GapItem that can be turned into a question for the user
or an LLM-driven suggestion for auto-resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.graph.builder import DependencyGraph
from src.ingestion.parser import ParseResult


class GapType(str, Enum):
    ORPHAN_COMPONENT = "orphan_component"
    ISOLATED_MODULE = "isolated_module"
    MISSING_DESCRIPTION = "missing_description"
    COUPLING_AMBIGUITY = "coupling_ambiguity"
    NO_API_SURFACE = "no_api_surface"
    NO_VARIANT_ASSIGNMENT = "no_variant_assignment"
    POTENTIAL_MISSING_LINK = "potential_missing_link"
    LOW_CONFIDENCE = "low_confidence"
    MISSING_CRITICALITY = "missing_criticality"


class GapSeverity(str, Enum):
    CRITICAL = "critical"  # Must be resolved before analysis is reliable
    HIGH = "high"          # Strongly recommended to resolve
    MEDIUM = "medium"      # Improves quality but not blocking
    LOW = "low"            # Nice to have


@dataclass
class GapItem:
    """A single identified gap in the knowledge graph."""

    gap_type: GapType
    severity: GapSeverity
    entity_id: str          # Component or module ID affected
    entity_name: str
    description: str        # Human-readable description of the gap
    question: str           # Question to ask the user to resolve this gap
    suggestion: str = ""    # LLM-suggested resolution (filled by GapAnalyzer)
    resolved: bool = False
    resolution: str = ""    # User's answer or auto-resolution

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_type": self.gap_type.value,
            "severity": self.severity.value,
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "description": self.description,
            "question": self.question,
            "suggestion": self.suggestion,
            "resolved": self.resolved,
            "resolution": self.resolution,
        }


@dataclass
class GapReport:
    """Complete gap analysis report."""

    gaps: list[GapItem] = field(default_factory=list)
    total_components: int = 0
    total_modules: int = 0

    @property
    def unresolved(self) -> list[GapItem]:
        return [g for g in self.gaps if not g.resolved]

    @property
    def critical_gaps(self) -> list[GapItem]:
        return [g for g in self.gaps if g.severity == GapSeverity.CRITICAL and not g.resolved]

    @property
    def completeness_score(self) -> float:
        """0-1 score of how complete the graph is (1 = no gaps)."""
        if not self.gaps:
            return 1.0
        resolved = sum(1 for g in self.gaps if g.resolved)
        return resolved / len(self.gaps)

    def summary(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for g in self.gaps:
            by_type[g.gap_type.value] = by_type.get(g.gap_type.value, 0) + 1
            by_severity[g.severity.value] = by_severity.get(g.severity.value, 0) + 1

        return {
            "total_gaps": len(self.gaps),
            "unresolved": len(self.unresolved),
            "critical": len(self.critical_gaps),
            "completeness_score": round(self.completeness_score, 2),
            "by_type": by_type,
            "by_severity": by_severity,
        }


class GapDetector:
    """Analyzes a DependencyGraph and ParseResult to find structural gaps."""

    def __init__(self, graph: DependencyGraph, parse_result: ParseResult | None = None) -> None:
        self._graph = graph
        self._parse_result = parse_result

    def detect(self) -> GapReport:
        """Run all gap detection checks and return a report."""
        from src.core.learning_narrator import narrate

        narrate("gaps.detect")
        report = GapReport(
            total_components=len(self._graph.get_components()),
            total_modules=self._graph.summary()["module_count"],
        )

        report.gaps.extend(self._check_orphan_components())
        report.gaps.extend(self._check_isolated_modules())
        report.gaps.extend(self._check_missing_descriptions())
        report.gaps.extend(self._check_coupling_ambiguity())
        report.gaps.extend(self._check_no_api_surface())
        report.gaps.extend(self._check_no_variant_assignment())
        report.gaps.extend(self._check_potential_missing_links())

        if self._parse_result:
            report.gaps.extend(self._check_low_confidence())

        # Sort by severity (critical first)
        severity_order = {
            GapSeverity.CRITICAL: 0,
            GapSeverity.HIGH: 1,
            GapSeverity.MEDIUM: 2,
            GapSeverity.LOW: 3,
        }
        report.gaps.sort(key=lambda g: severity_order[g.severity])

        return report

    def _check_orphan_components(self) -> list[GapItem]:
        """Find components with no module connections."""
        gaps: list[GapItem] = []
        for comp_id, comp in self._graph.get_components().items():
            modules = self._graph.get_component_modules(comp_id)
            if not modules:
                gaps.append(GapItem(
                    gap_type=GapType.ORPHAN_COMPONENT,
                    severity=GapSeverity.CRITICAL,
                    entity_id=comp_id,
                    entity_name=comp.name,
                    description=(
                        f"Component '{comp.name}' has no module connections. "
                        "It cannot participate in impact analysis."
                    ),
                    question=(
                        f"Component '{comp.name}' appears isolated. "
                        "What shared libraries, services, or modules does it depend on? "
                        "For example: databases, message queues, shared utilities, config services."
                    ),
                ))
        return gaps

    def _check_isolated_modules(self) -> list[GapItem]:
        """Find modules connected to only one component."""
        gaps: list[GapItem] = []
        graph = self._graph.graph

        for node, data in graph.nodes(data=True):
            if data.get("node_type") != "module":
                continue
            components = self._graph.get_module_components(node)
            if len(components) == 1:
                gaps.append(GapItem(
                    gap_type=GapType.ISOLATED_MODULE,
                    severity=GapSeverity.MEDIUM,
                    entity_id=node,
                    entity_name=node,
                    description=(
                        f"Module '{node}' is only used by {components[0]}. "
                        "It won't create cross-component impact paths."
                    ),
                    question=(
                        f"Module '{node}' is currently only linked to '{components[0]}'. "
                        "Are there other components or services that also use this module?"
                    ),
                ))
        return gaps

    def _check_missing_descriptions(self) -> list[GapItem]:
        """Find components with empty or generic descriptions."""
        gaps: list[GapItem] = []
        generic_prefixes = ["Service:", "Class ", "Directory module:"]

        for comp_id, comp in self._graph.get_components().items():
            desc = comp.description.strip()
            is_generic = any(desc.startswith(p) for p in generic_prefixes)

            if not desc or is_generic:
                gaps.append(GapItem(
                    gap_type=GapType.MISSING_DESCRIPTION,
                    severity=GapSeverity.HIGH,
                    entity_id=comp_id,
                    entity_name=comp.name,
                    description=(
                        f"Component '{comp.name}' has a missing or auto-generated description. "
                        "This weakens semantic search accuracy."
                    ),
                    question=(
                        f"What does '{comp.name}' do? Please provide a 1-2 sentence description "
                        "of its purpose, key responsibilities, and critical data it processes."
                    ),
                ))
        return gaps

    def _check_coupling_ambiguity(self) -> list[GapItem]:
        """Flag components where all module couplings are the same default."""
        gaps: list[GapItem] = []
        for comp_id, comp in self._graph.get_components().items():
            modules = self._graph.get_component_modules(comp_id)
            if len(modules) < 2:
                continue

            couplings = {m["coupling"] for m in modules}
            if len(couplings) == 1 and "tight" in couplings:
                mod_names = ", ".join(m["module_id"] for m in modules)
                gaps.append(GapItem(
                    gap_type=GapType.COUPLING_AMBIGUITY,
                    severity=GapSeverity.MEDIUM,
                    entity_id=comp_id,
                    entity_name=comp.name,
                    description=(
                        f"All {len(modules)} modules for '{comp.name}' are marked as 'tight' coupling. "
                        "This may be a default from auto-detection."
                    ),
                    question=(
                        f"Component '{comp.name}' uses these modules: {mod_names}. "
                        "For each, is the coupling 'tight' (critical dependency), "
                        "'loose' (used but could be swapped), or 'optional' (feature-flagged or degradable)?"
                    ),
                ))
        return gaps

    def _check_no_api_surface(self) -> list[GapItem]:
        """Find components with no documented API surface."""
        gaps: list[GapItem] = []
        for comp_id, comp in self._graph.get_components().items():
            if not comp.api_surface:
                gaps.append(GapItem(
                    gap_type=GapType.NO_API_SURFACE,
                    severity=GapSeverity.LOW,
                    entity_id=comp_id,
                    entity_name=comp.name,
                    description=(
                        f"Component '{comp.name}' has no documented API endpoints. "
                        "This reduces search specificity."
                    ),
                    question=(
                        f"What API endpoints or interfaces does '{comp.name}' expose? "
                        "List any REST endpoints, gRPC services, event handlers, or CLI commands."
                    ),
                ))
        return gaps

    def _check_no_variant_assignment(self) -> list[GapItem]:
        """Find components not assigned to any product variant."""
        gaps: list[GapItem] = []
        variants = self._graph.get_variants()

        if not variants:
            return gaps  # No variants defined yet — not a gap

        for comp_id, comp in self._graph.get_components().items():
            variant_ids = self._graph.get_variants_for_component(comp_id)
            if not variant_ids:
                gaps.append(GapItem(
                    gap_type=GapType.NO_VARIANT_ASSIGNMENT,
                    severity=GapSeverity.MEDIUM,
                    entity_id=comp_id,
                    entity_name=comp.name,
                    description=(
                        f"Component '{comp.name}' is not included in any product variant."
                    ),
                    question=(
                        f"Which product variants or deployment tiers include '{comp.name}'? "
                        "For example: starter, professional, enterprise, or specific regions."
                    ),
                ))
        return gaps

    def _check_potential_missing_links(self) -> list[GapItem]:
        """Detect components that likely should be connected but aren't.

        Heuristic: components in the same source directory that share no modules.
        """
        gaps: list[GapItem] = []
        if not self._parse_result:
            return gaps

        # Group components by their source directory
        dir_components: dict[str, list[str]] = {}
        for comp in self._parse_result.components:
            if "/" in comp.source_file:
                parent = comp.source_file.rsplit("/", 1)[0]
            else:
                parent = "."
            if parent not in dir_components:
                dir_components[parent] = []
            dir_components[parent].append(comp.id)

        # Check each group for unconnected pairs
        for parent_dir, comp_ids in dir_components.items():
            if len(comp_ids) < 2:
                continue

            for i, c1 in enumerate(comp_ids):
                for c2 in comp_ids[i + 1:]:
                    mods_1 = {m["module_id"] for m in self._graph.get_component_modules(c1)}
                    mods_2 = {m["module_id"] for m in self._graph.get_component_modules(c2)}
                    shared = mods_1 & mods_2

                    if not shared and mods_1 and mods_2:
                        gaps.append(GapItem(
                            gap_type=GapType.POTENTIAL_MISSING_LINK,
                            severity=GapSeverity.LOW,
                            entity_id=f"{c1}:{c2}",
                            entity_name=f"{c1} <-> {c2}",
                            description=(
                                f"Components '{c1}' and '{c2}' are in the same directory "
                                f"({parent_dir}) but share no modules."
                            ),
                            question=(
                                f"Components '{c1}' and '{c2}' coexist in {parent_dir}. "
                                "Do they share any dependencies, communicate via events, "
                                "or use any common infrastructure?"
                            ),
                        ))
        return gaps

    def _check_low_confidence(self) -> list[GapItem]:
        """Flag components with low detection confidence."""
        gaps: list[GapItem] = []
        if not self._parse_result:
            return gaps

        for comp in self._parse_result.components:
            if comp.confidence < 0.5:
                gaps.append(GapItem(
                    gap_type=GapType.LOW_CONFIDENCE,
                    severity=GapSeverity.HIGH,
                    entity_id=comp.id,
                    entity_name=comp.name,
                    description=(
                        f"Component '{comp.name}' was detected with low confidence "
                        f"({comp.confidence:.0%}). It may not be a real component."
                    ),
                    question=(
                        f"Is '{comp.name}' (from {comp.source_file}) a real component/service "
                        "in your system, or was it incorrectly detected? "
                        "If real, what does it do?"
                    ),
                ))
        return gaps
