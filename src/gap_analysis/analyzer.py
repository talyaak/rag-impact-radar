"""Interactive gap analyzer — uses LLM to suggest resolutions and refines the graph.

This module integrates the GapDetector with the LLM to:
  1. Analyze detected gaps and generate intelligent suggestions
  2. Present curated questions to the user (via CLI or API)
  3. Apply user answers to refine the dependency graph
  4. Re-run detection until the graph reaches acceptable completeness

The analyzer maintains a session state so the refinement loop can be
paused and resumed across API calls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.graph.builder import DependencyGraph
from src.core.llm_client import LLMClient
from src.gap_analysis.detector import GapDetector, GapItem, GapReport, GapSeverity, GapType
from src.ingestion.parser import ParseResult

logger = logging.getLogger(__name__)

# System prompt for gap analysis LLM calls
_GAP_SYSTEM_PROMPT = (
    "You are a senior software architect analyzing a dependency graph for completeness. "
    "Your job is to suggest reasonable defaults for missing information based on common "
    "software patterns. Be concise and specific. Do not speculate beyond what the evidence "
    "supports. Format your response as JSON when asked."
)


@dataclass
class GapQuestion:
    """A question to present to the user during the onboarding loop."""

    gap_index: int
    gap_type: str
    entity_id: str
    entity_name: str
    question: str
    suggestion: str
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_index": self.gap_index,
            "gap_type": self.gap_type,
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "question": self.question,
            "suggestion": self.suggestion,
            "severity": self.severity,
        }


@dataclass
class GapAnswer:
    """User's answer to a gap question."""

    gap_index: int
    answer: str
    accept_suggestion: bool = False


@dataclass
class AnalysisSession:
    """Tracks the state of an interactive gap analysis session."""

    session_id: str
    gap_report: GapReport
    pending_questions: list[GapQuestion] = field(default_factory=list)
    answered_questions: list[tuple[GapQuestion, GapAnswer]] = field(default_factory=list)
    iteration: int = 0
    is_complete: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "iteration": self.iteration,
            "total_gaps": len(self.gap_report.gaps),
            "pending_questions": len(self.pending_questions),
            "answered_questions": len(self.answered_questions),
            "completeness_score": self.gap_report.completeness_score,
            "is_complete": self.is_complete,
        }


class GapAnalyzer:
    """Orchestrates the interactive LLM-driven gap analysis loop.

    The analyzer follows this flow:
      1. Detect gaps in the current graph
      2. Use LLM to generate suggestions for each gap
      3. Present questions to the user (prioritized by severity)
      4. Apply answers to refine the graph
      5. Re-detect to find remaining gaps
      6. Repeat until completeness threshold is met
    """

    def __init__(
        self,
        graph: DependencyGraph,
        llm_client: LLMClient | None = None,
        parse_result: ParseResult | None = None,
        config_path: str | Path = "config/model_config.yaml",
        completeness_threshold: float = 0.8,
    ) -> None:
        self._graph = graph
        self._llm = llm_client
        self._parse_result = parse_result
        self._config = self._load_config(config_path)
        self._threshold = completeness_threshold

        gap_config = self._config.get("gap_analysis", {})
        self._max_questions_per_round = gap_config.get("max_questions_per_round", 10)
        self._auto_resolve_low = gap_config.get("auto_resolve_low_severity", True)

    @staticmethod
    def _load_config(config_path: str | Path) -> dict[str, Any]:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def start_session(self, session_id: str = "default") -> AnalysisSession:
        """Start a new gap analysis session.

        Detects all gaps, generates LLM suggestions (if available),
        and returns the session with prioritized questions.
        """
        detector = GapDetector(self._graph, self._parse_result)
        gap_report = detector.detect()

        logger.info(
            "Gap detection complete: %d gaps found (%d critical)",
            len(gap_report.gaps),
            len(gap_report.critical_gaps),
        )

        # Generate LLM suggestions for gaps
        if self._llm is not None:
            self._generate_suggestions(gap_report)

        # Auto-resolve low severity gaps if configured
        if self._auto_resolve_low:
            self._auto_resolve(gap_report)

        # Build questions from unresolved gaps
        questions = self._build_questions(gap_report)

        session = AnalysisSession(
            session_id=session_id,
            gap_report=gap_report,
            pending_questions=questions,
        )

        if not questions:
            session.is_complete = True

        return session

    def apply_answers(
        self,
        session: AnalysisSession,
        answers: list[GapAnswer],
    ) -> AnalysisSession:
        """Apply user answers to refine the graph and re-detect gaps.

        Args:
            session: The current analysis session.
            answers: List of user answers to pending questions.

        Returns:
            Updated session with new gap report and questions.
        """
        # Apply each answer
        for answer in answers:
            if answer.gap_index >= len(session.gap_report.gaps):
                continue

            gap = session.gap_report.gaps[answer.gap_index]

            if answer.accept_suggestion and gap.suggestion:
                resolution = gap.suggestion
            else:
                resolution = answer.answer

            # Apply the resolution to the graph
            self._apply_resolution(gap, resolution)
            gap.resolved = True
            gap.resolution = resolution

            # Track answered questions
            matching_q = [
                q for q in session.pending_questions
                if q.gap_index == answer.gap_index
            ]
            if matching_q:
                session.answered_questions.append((matching_q[0], answer))

        # Re-detect gaps
        session.iteration += 1
        detector = GapDetector(self._graph, self._parse_result)
        new_report = detector.detect()

        # Merge resolved status from previous report
        old_resolved = {g.entity_id: g for g in session.gap_report.gaps if g.resolved}
        for gap in new_report.gaps:
            if gap.entity_id in old_resolved:
                gap.resolved = True
                gap.resolution = old_resolved[gap.entity_id].resolution

        session.gap_report = new_report

        # Check completeness
        if new_report.completeness_score >= self._threshold:
            session.is_complete = True
            session.pending_questions = []
        else:
            # Generate new questions for remaining gaps
            if self._llm is not None:
                self._generate_suggestions(new_report)
            session.pending_questions = self._build_questions(new_report)

        return session

    def _generate_suggestions(self, gap_report: GapReport) -> None:
        """Use LLM to generate resolution suggestions for unresolved gaps."""
        if self._llm is None:
            return

        unresolved = gap_report.unresolved
        if not unresolved:
            return

        # Build context about the current graph state
        graph_context = self._build_graph_context()

        # Generate suggestions in batches
        for gap in unresolved:
            prompt = self._build_suggestion_prompt(gap, graph_context)
            try:
                suggestion = self._llm.generate(
                    prompt=prompt,
                    system_prompt=_GAP_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_tokens=256,
                )
                gap.suggestion = suggestion.strip()
            except Exception:
                logger.warning(
                    "Failed to generate suggestion for gap %s",
                    gap.entity_id,
                    exc_info=True,
                )

    def _build_graph_context(self) -> str:
        """Build a summary of the current graph for LLM context."""
        components = self._graph.get_components()
        lines = ["Current graph state:"]
        lines.append(f"- {len(components)} components")
        lines.append(f"- {self._graph.summary()['module_count']} modules")
        lines.append(f"- {self._graph.summary()['edge_count']} edges")
        lines.append("")
        lines.append("Components:")
        for comp_id, comp in sorted(components.items()):
            modules = self._graph.get_component_modules(comp_id)
            mod_names = ", ".join(m["module_id"] for m in modules) if modules else "(none)"
            lines.append(f"  - {comp.name} [{comp.criticality}]: {mod_names}")
        return "\n".join(lines)

    def _build_suggestion_prompt(self, gap: GapItem, graph_context: str) -> str:
        """Build a prompt asking the LLM to suggest a resolution for a gap."""
        return (
            f"{graph_context}\n\n"
            f"Gap detected ({gap.gap_type.value}, severity={gap.severity.value}):\n"
            f"Entity: {gap.entity_name} ({gap.entity_id})\n"
            f"Issue: {gap.description}\n\n"
            f"Based on common software architecture patterns and the graph context above, "
            f"suggest a reasonable resolution. Be specific and concise (1-2 sentences). "
            f"If the gap is about modules, suggest specific module names that likely connect "
            f"this component to others. If about coupling, suggest appropriate coupling levels."
        )

    def _auto_resolve(self, gap_report: GapReport) -> None:
        """Auto-resolve low-severity gaps with reasonable defaults."""
        for gap in gap_report.gaps:
            if gap.resolved:
                continue
            if gap.severity != GapSeverity.LOW:
                continue

            if gap.gap_type == GapType.NO_API_SURFACE:
                gap.resolved = True
                gap.resolution = "No API surface documented (acceptable for internal components)"
            elif gap.gap_type == GapType.POTENTIAL_MISSING_LINK:
                if gap.suggestion:
                    gap.resolved = True
                    gap.resolution = gap.suggestion

    def _build_questions(self, gap_report: GapReport) -> list[GapQuestion]:
        """Build prioritized questions from unresolved gaps."""
        questions: list[GapQuestion] = []
        unresolved = gap_report.unresolved

        for i, gap in enumerate(unresolved):
            if len(questions) >= self._max_questions_per_round:
                break

            questions.append(GapQuestion(
                gap_index=self._gap_report_index(gap_report, gap),
                gap_type=gap.gap_type.value,
                entity_id=gap.entity_id,
                entity_name=gap.entity_name,
                question=gap.question,
                suggestion=gap.suggestion,
                severity=gap.severity.value,
            ))

        return questions

    def _gap_report_index(self, report: GapReport, gap: GapItem) -> int:
        """Find the index of a gap in the report's gaps list."""
        for i, g in enumerate(report.gaps):
            if g is gap:
                return i
        return -1

    def _apply_resolution(self, gap: GapItem, resolution: str) -> None:
        """Apply a resolution to the dependency graph.

        Interprets the resolution text and makes the appropriate graph mutation.
        """
        if gap.gap_type == GapType.ORPHAN_COMPONENT:
            self._resolve_orphan(gap, resolution)
        elif gap.gap_type == GapType.MISSING_DESCRIPTION:
            self._resolve_description(gap, resolution)
        elif gap.gap_type == GapType.COUPLING_AMBIGUITY:
            self._resolve_coupling(gap, resolution)
        elif gap.gap_type == GapType.ISOLATED_MODULE:
            self._resolve_isolated_module(gap, resolution)
        elif gap.gap_type == GapType.MISSING_CRITICALITY:
            self._resolve_criticality(gap, resolution)
        # Other gap types are informational and don't mutate the graph

    def _resolve_orphan(self, gap: GapItem, resolution: str) -> None:
        """Add module connections for an orphan component."""
        # Try to parse module names from the resolution
        # Look for common patterns: comma-separated, bulleted, etc.
        module_names = self._extract_module_names(resolution)
        graph = self._graph.graph

        for mod_name in module_names:
            mod_id = mod_name.strip().lower().replace(" ", "_").replace("-", "_")
            if not graph.has_node(mod_id):
                graph.add_node(mod_id, node_type="module")
            if not graph.has_edge(gap.entity_id, mod_id):
                graph.add_edge(
                    gap.entity_id,
                    mod_id,
                    coupling="tight",
                    usage=f"Resolved via gap analysis: {resolution[:100]}",
                )

    def _resolve_description(self, gap: GapItem, resolution: str) -> None:
        """Update a component's description."""
        graph = self._graph.graph
        if graph.has_node(gap.entity_id):
            graph.nodes[gap.entity_id]["description"] = resolution

        comp = self._graph.get_component(gap.entity_id)
        if comp:
            # Update the ComponentData object directly
            object.__setattr__(comp, "description", resolution)

    def _resolve_coupling(self, gap: GapItem, resolution: str) -> None:
        """Update coupling strengths based on resolution."""
        # Try to parse coupling assignments from resolution
        resolution_lower = resolution.lower()
        graph = self._graph.graph
        modules = self._graph.get_component_modules(gap.entity_id)

        for mod in modules:
            mod_id = mod["module_id"]
            if mod_id in resolution_lower:
                # Check for coupling keywords near the module name
                idx = resolution_lower.index(mod_id)
                context = resolution_lower[max(0, idx - 30):idx + len(mod_id) + 30]
                if "optional" in context:
                    coupling = "optional"
                elif "loose" in context:
                    coupling = "loose"
                else:
                    coupling = "tight"

                if graph.has_edge(gap.entity_id, mod_id):
                    graph.edges[gap.entity_id, mod_id]["coupling"] = coupling

    def _resolve_isolated_module(self, gap: GapItem, resolution: str) -> None:
        """Add component connections for an isolated module."""
        component_names = self._extract_module_names(resolution)
        graph = self._graph.graph

        for comp_name in component_names:
            comp_id = comp_name.strip().lower().replace(" ", "_").replace("-", "_")
            if graph.has_node(comp_id) and graph.nodes[comp_id].get("node_type") == "component":
                if not graph.has_edge(comp_id, gap.entity_id):
                    graph.add_edge(
                        comp_id,
                        gap.entity_id,
                        coupling="loose",
                        usage=f"Resolved via gap analysis: {resolution[:100]}",
                    )

    def _resolve_criticality(self, gap: GapItem, resolution: str) -> None:
        """Update a component's criticality level."""
        resolution_lower = resolution.lower().strip()
        graph = self._graph.graph

        for level in ("critical", "high", "medium", "low"):
            if level in resolution_lower:
                if graph.has_node(gap.entity_id):
                    graph.nodes[gap.entity_id]["criticality"] = level
                break

    def _extract_module_names(self, text: str) -> list[str]:
        """Extract module/component names from free-form text."""
        import re
        # Try JSON array first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (json.JSONDecodeError, TypeError):
            pass

        # Try comma-separated
        if "," in text:
            candidates = [s.strip().strip(".-") for s in text.split(",")]
            return [c for c in candidates if c and len(c) < 60 and " " not in c or "_" in c]

        # Try line-by-line (bullet points)
        lines = text.strip().split("\n")
        names: list[str] = []
        for line in lines:
            line = line.strip().lstrip("-*•").strip()
            # Extract the first word/identifier
            match = re.match(r"^[\w_-]+", line)
            if match and len(match.group()) > 2:
                names.append(match.group())

        return names if names else [text.strip()[:50]]
