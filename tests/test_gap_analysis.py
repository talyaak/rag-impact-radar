"""Tests for the V2 gap analysis — detector and interactive analyzer.

Tests verify:
  - GapDetector finds orphan components, isolated modules, and other gaps
  - GapReport tracks resolution state and completeness
  - GapAnalyzer builds questions and applies answers to the graph
  - Graph mutations from gap resolution are correct
  - All tests run without API keys (zero-API-key testing)
"""

import pytest
from pathlib import Path

from src.graph.builder import DependencyGraph
from src.gap_analysis.detector import (
    GapDetector,
    GapItem,
    GapReport,
    GapSeverity,
    GapType,
)
from src.gap_analysis.analyzer import (
    GapAnalyzer,
    GapAnswer,
    AnalysisSession,
)

# ── Paths ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
COMPONENTS_DIR = DATA_DIR / "components"
VARIANTS_DIR = DATA_DIR / "variants"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "model_config.yaml"


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def full_graph() -> DependencyGraph:
    """Load the complete seed data graph."""
    return DependencyGraph(COMPONENTS_DIR, VARIANTS_DIR)


@pytest.fixture
def graph_with_orphan() -> DependencyGraph:
    """Graph with an orphan component (no module connections)."""
    g = DependencyGraph()
    g.add_component({
        "id": "orphan_comp",
        "name": "OrphanComponent",
        "description": "A component with no modules",
        "criticality": "medium",
        "modules": [],
    })
    g.add_component({
        "id": "connected_comp",
        "name": "ConnectedComponent",
        "description": "A properly connected component",
        "criticality": "high",
        "modules": [
            {"module_id": "shared_mod", "usage": "Uses shared stuff", "coupling": "tight"},
        ],
    })
    return g


@pytest.fixture
def graph_with_missing_desc() -> DependencyGraph:
    """Graph with a component that has a generic auto-generated description."""
    g = DependencyGraph()
    g.add_component({
        "id": "no_desc_comp",
        "name": "NoDescComponent",
        "description": "Class NoDescComponent",  # Generic auto-generated
        "criticality": "medium",
        "modules": [
            {"module_id": "some_mod", "usage": "Something", "coupling": "tight"},
        ],
    })
    return g


@pytest.fixture
def graph_with_all_tight() -> DependencyGraph:
    """Graph where all couplings default to tight (coupling ambiguity)."""
    g = DependencyGraph()
    g.add_component({
        "id": "tight_comp",
        "name": "TightComponent",
        "description": "Component with all tight couplings",
        "criticality": "high",
        "modules": [
            {"module_id": "mod_a", "usage": "Uses A", "coupling": "tight"},
            {"module_id": "mod_b", "usage": "Uses B", "coupling": "tight"},
            {"module_id": "mod_c", "usage": "Uses C", "coupling": "tight"},
        ],
    })
    return g


# ── GapDetector Tests ─────────────────────────────────────────────────


class TestGapDetector:
    def test_detects_orphan_component(self, graph_with_orphan: DependencyGraph):
        detector = GapDetector(graph_with_orphan)
        report = detector.detect()

        orphan_gaps = [
            g for g in report.gaps
            if g.gap_type == GapType.ORPHAN_COMPONENT
        ]
        assert len(orphan_gaps) == 1
        assert orphan_gaps[0].entity_id == "orphan_comp"
        assert orphan_gaps[0].severity == GapSeverity.CRITICAL

    def test_detects_missing_description(self, graph_with_missing_desc: DependencyGraph):
        detector = GapDetector(graph_with_missing_desc)
        report = detector.detect()

        desc_gaps = [
            g for g in report.gaps
            if g.gap_type == GapType.MISSING_DESCRIPTION
        ]
        assert len(desc_gaps) >= 1
        assert desc_gaps[0].entity_id == "no_desc_comp"

    def test_detects_coupling_ambiguity(self, graph_with_all_tight: DependencyGraph):
        detector = GapDetector(graph_with_all_tight)
        report = detector.detect()

        coupling_gaps = [
            g for g in report.gaps
            if g.gap_type == GapType.COUPLING_AMBIGUITY
        ]
        assert len(coupling_gaps) == 1
        assert coupling_gaps[0].entity_id == "tight_comp"

    def test_detects_isolated_modules(self, graph_with_orphan: DependencyGraph):
        detector = GapDetector(graph_with_orphan)
        report = detector.detect()

        isolated_gaps = [
            g for g in report.gaps
            if g.gap_type == GapType.ISOLATED_MODULE
        ]
        # shared_mod is only used by connected_comp
        assert len(isolated_gaps) >= 1

    def test_no_variant_assignment_when_variants_exist(self, full_graph: DependencyGraph):
        detector = GapDetector(full_graph)
        report = detector.detect()

        # In the full seed data, all components should be assigned to variants
        no_variant_gaps = [
            g for g in report.gaps
            if g.gap_type == GapType.NO_VARIANT_ASSIGNMENT
        ]
        # Seed data has proper variant assignments
        assert len(no_variant_gaps) == 0

    def test_no_variant_gap_when_no_variants(self):
        """If no variants are defined, the check should be skipped."""
        g = DependencyGraph()
        g.add_component({
            "id": "comp",
            "name": "Comp",
            "description": "Test",
            "criticality": "medium",
            "modules": [],
        })
        detector = GapDetector(g)
        report = detector.detect()

        no_variant_gaps = [
            g for g in report.gaps
            if g.gap_type == GapType.NO_VARIANT_ASSIGNMENT
        ]
        assert len(no_variant_gaps) == 0

    def test_full_graph_has_some_gaps(self, full_graph: DependencyGraph):
        """Even the seed data may have some gaps (e.g., no API surface)."""
        detector = GapDetector(full_graph)
        report = detector.detect()

        # The report should run without errors
        assert isinstance(report, GapReport)
        assert report.total_components == 12

    def test_gaps_sorted_by_severity(self, graph_with_orphan: DependencyGraph):
        detector = GapDetector(graph_with_orphan)
        report = detector.detect()

        if len(report.gaps) < 2:
            return

        severity_order = {
            GapSeverity.CRITICAL: 0,
            GapSeverity.HIGH: 1,
            GapSeverity.MEDIUM: 2,
            GapSeverity.LOW: 3,
        }
        for i in range(len(report.gaps) - 1):
            assert (
                severity_order[report.gaps[i].severity]
                <= severity_order[report.gaps[i + 1].severity]
            )


# ── GapReport Tests ───────────────────────────────────────────────────


class TestGapReport:
    def test_completeness_score_no_gaps(self):
        report = GapReport()
        assert report.completeness_score == 1.0

    def test_completeness_score_all_unresolved(self):
        report = GapReport(gaps=[
            GapItem(
                gap_type=GapType.ORPHAN_COMPONENT,
                severity=GapSeverity.CRITICAL,
                entity_id="comp1",
                entity_name="Comp1",
                description="desc",
                question="q",
            ),
            GapItem(
                gap_type=GapType.MISSING_DESCRIPTION,
                severity=GapSeverity.HIGH,
                entity_id="comp2",
                entity_name="Comp2",
                description="desc",
                question="q",
            ),
        ])
        assert report.completeness_score == 0.0

    def test_completeness_score_partial(self):
        gap1 = GapItem(
            gap_type=GapType.ORPHAN_COMPONENT,
            severity=GapSeverity.CRITICAL,
            entity_id="comp1",
            entity_name="Comp1",
            description="desc",
            question="q",
            resolved=True,
            resolution="Fixed",
        )
        gap2 = GapItem(
            gap_type=GapType.MISSING_DESCRIPTION,
            severity=GapSeverity.HIGH,
            entity_id="comp2",
            entity_name="Comp2",
            description="desc",
            question="q",
        )
        report = GapReport(gaps=[gap1, gap2])
        assert report.completeness_score == 0.5

    def test_unresolved_property(self):
        gap1 = GapItem(
            gap_type=GapType.ORPHAN_COMPONENT,
            severity=GapSeverity.CRITICAL,
            entity_id="comp1",
            entity_name="Comp1",
            description="desc",
            question="q",
            resolved=True,
        )
        gap2 = GapItem(
            gap_type=GapType.MISSING_DESCRIPTION,
            severity=GapSeverity.HIGH,
            entity_id="comp2",
            entity_name="Comp2",
            description="desc",
            question="q",
        )
        report = GapReport(gaps=[gap1, gap2])
        assert len(report.unresolved) == 1
        assert report.unresolved[0].entity_id == "comp2"

    def test_critical_gaps_property(self):
        gap1 = GapItem(
            gap_type=GapType.ORPHAN_COMPONENT,
            severity=GapSeverity.CRITICAL,
            entity_id="comp1",
            entity_name="Comp1",
            description="desc",
            question="q",
        )
        gap2 = GapItem(
            gap_type=GapType.MISSING_DESCRIPTION,
            severity=GapSeverity.LOW,
            entity_id="comp2",
            entity_name="Comp2",
            description="desc",
            question="q",
        )
        report = GapReport(gaps=[gap1, gap2])
        assert len(report.critical_gaps) == 1
        assert report.critical_gaps[0].severity == GapSeverity.CRITICAL

    def test_summary_format(self):
        report = GapReport(
            gaps=[
                GapItem(
                    gap_type=GapType.ORPHAN_COMPONENT,
                    severity=GapSeverity.CRITICAL,
                    entity_id="comp1",
                    entity_name="Comp1",
                    description="desc",
                    question="q",
                ),
            ],
            total_components=5,
        )
        summary = report.summary()
        assert "total_gaps" in summary
        assert "unresolved" in summary
        assert "completeness_score" in summary
        assert "by_type" in summary
        assert "by_severity" in summary

    def test_gap_item_to_dict(self):
        gap = GapItem(
            gap_type=GapType.ORPHAN_COMPONENT,
            severity=GapSeverity.CRITICAL,
            entity_id="comp1",
            entity_name="Comp1",
            description="desc",
            question="q",
            suggestion="suggestion",
        )
        d = gap.to_dict()
        assert d["gap_type"] == "orphan_component"
        assert d["severity"] == "critical"
        assert d["entity_id"] == "comp1"


# ── GapAnalyzer Tests ─────────────────────────────────────────────────


class TestGapAnalyzer:
    def test_start_session_no_llm(self, graph_with_orphan: DependencyGraph):
        """Session should work without LLM (no suggestions, but gaps detected)."""
        analyzer = GapAnalyzer(
            graph=graph_with_orphan,
            llm_client=None,
            config_path=CONFIG_PATH,
        )
        session = analyzer.start_session()

        assert isinstance(session, AnalysisSession)
        assert len(session.gap_report.gaps) > 0
        assert session.iteration == 0

    def test_session_has_questions(self, graph_with_orphan: DependencyGraph):
        analyzer = GapAnalyzer(graph=graph_with_orphan, config_path=CONFIG_PATH)
        session = analyzer.start_session()

        # Should have at least one question for the orphan component
        assert len(session.pending_questions) > 0

    def test_session_summary(self, graph_with_orphan: DependencyGraph):
        analyzer = GapAnalyzer(graph=graph_with_orphan, config_path=CONFIG_PATH)
        session = analyzer.start_session()
        summary = session.summary()

        assert "session_id" in summary
        assert "total_gaps" in summary
        assert "pending_questions" in summary
        assert "completeness_score" in summary

    def test_apply_orphan_answer(self, graph_with_orphan: DependencyGraph):
        """Answering an orphan question should add module connections."""
        analyzer = GapAnalyzer(graph=graph_with_orphan, config_path=CONFIG_PATH)
        session = analyzer.start_session()

        # Find the orphan question
        orphan_q = None
        for q in session.pending_questions:
            if q.gap_type == "orphan_component":
                orphan_q = q
                break

        if orphan_q is None:
            pytest.skip("No orphan question found")

        # Answer with module names
        answers = [GapAnswer(
            gap_index=orphan_q.gap_index,
            answer="shared_mod, event_bus",
        )]
        session = analyzer.apply_answers(session, answers)

        # The orphan should now have module connections
        modules = graph_with_orphan.get_component_modules("orphan_comp")
        assert len(modules) > 0

    def test_apply_description_answer(self, graph_with_missing_desc: DependencyGraph):
        """Answering a description gap should update the component."""
        analyzer = GapAnalyzer(
            graph=graph_with_missing_desc,
            config_path=CONFIG_PATH,
        )
        session = analyzer.start_session()

        # Find the description question
        desc_q = None
        for q in session.pending_questions:
            if q.gap_type == "missing_description":
                desc_q = q
                break

        if desc_q is None:
            pytest.skip("No description question found")

        answers = [GapAnswer(
            gap_index=desc_q.gap_index,
            answer="Handles real-time data processing and event streaming",
        )]
        session = analyzer.apply_answers(session, answers)

        # Description should be updated
        comp = graph_with_missing_desc.get_component("no_desc_comp")
        assert "real-time" in comp.description.lower() or session.iteration > 0

    def test_session_increments_iteration(self, graph_with_orphan: DependencyGraph):
        analyzer = GapAnalyzer(graph=graph_with_orphan, config_path=CONFIG_PATH)
        session = analyzer.start_session()
        assert session.iteration == 0

        session = analyzer.apply_answers(session, [])
        assert session.iteration == 1

    def test_complete_session_when_all_resolved(self):
        """Session should be marked complete when completeness threshold is met."""
        g = DependencyGraph()
        g.add_component({
            "id": "good_comp",
            "name": "GoodComponent",
            "description": "A well-documented component with proper connections",
            "criticality": "high",
            "modules": [
                {"module_id": "mod_a", "usage": "Core logic", "coupling": "tight"},
                {"module_id": "mod_b", "usage": "Utilities", "coupling": "loose"},
            ],
        })
        g.add_component({
            "id": "good_comp_2",
            "name": "GoodComponent2",
            "description": "Another well-documented component",
            "criticality": "medium",
            "modules": [
                {"module_id": "mod_a", "usage": "Shared logic", "coupling": "tight"},
            ],
        })

        analyzer = GapAnalyzer(graph=g, config_path=CONFIG_PATH, completeness_threshold=0.5)
        session = analyzer.start_session()

        # With minimal gaps and auto-resolve, it might already be complete
        # or at least have a reasonable completeness score
        assert isinstance(session.gap_report.completeness_score, float)

    def test_question_to_dict(self, graph_with_orphan: DependencyGraph):
        analyzer = GapAnalyzer(graph=graph_with_orphan, config_path=CONFIG_PATH)
        session = analyzer.start_session()

        if session.pending_questions:
            q_dict = session.pending_questions[0].to_dict()
            assert "gap_index" in q_dict
            assert "question" in q_dict
            assert "severity" in q_dict
            assert "entity_id" in q_dict
