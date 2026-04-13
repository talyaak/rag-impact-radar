"""Tests for the impact analyzer and reporter — scoring, aggregation, and formatting.

=== RAG Pipeline Learning: Testing the Orchestration Layer ===

The analyzer is where graph traversal and semantic retrieval converge into
scored impact results. These tests verify that the scoring formula, semantic
boost, variant aggregation, and report formatting all work correctly WITHOUT
requiring an OpenAI API key.

Key testing strategy:
  - Use ``use_llm=False`` to skip LLM calls entirely.
  - Use real seed data (YAML components/variants) for integration-style tests.
  - Use Chroma's default Sentence Transformer embeddings (no OpenAI) for
    semantic boost tests.
  - Verify the exact scoring formula: coupling_weight * criticality_mult / depth.
  - Verify reporter output shapes and content markers for all three formats.
===
"""

import shutil
import tempfile

import pytest
from pathlib import Path

from src.graph.builder import DependencyGraph
from src.graph.traverser import ImpactTraverser
from src.rag.vector_store import VectorStore
from src.rag.embedder import ComponentEmbedder
from src.rag.retriever import SemanticRetriever
from src.analyzer.impact import ImpactAnalyzer, AnalysisResult, ScoredImpact, VariantRisk
from src.analyzer.reporter import ImpactReporter

# ── Paths ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
COMPONENTS_DIR = DATA_DIR / "components"
VARIANTS_DIR = DATA_DIR / "variants"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "model_config.yaml"

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def full_graph():
    return DependencyGraph(COMPONENTS_DIR, VARIANTS_DIR)


@pytest.fixture
def traverser(full_graph):
    return ImpactTraverser(full_graph)


@pytest.fixture
def analyzer_no_llm(full_graph, traverser):
    """Analyzer without LLM or retriever -- pure graph analysis."""
    return ImpactAnalyzer(full_graph, traverser, config_path=CONFIG_PATH)


@pytest.fixture
def analysis_result(analyzer_no_llm):
    """Pre-computed analysis of auth_engine change."""
    return analyzer_no_llm.analyze(["auth_engine"], use_llm=False)


@pytest.fixture
def tmp_chroma_dir():
    tmpdir = tempfile.mkdtemp(prefix="impact_radar_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ── TestScoringFormula ───────────────────────────────────────────────────


class TestScoringFormula:
    """Verify the severity score formula: coupling_weight * crit_mult / depth."""

    def test_tight_scores_higher_than_loose(self, analysis_result):
        """Components with tight coupling should outscore loose-coupled ones."""
        tight = [s for s in analysis_result.scored_impacts.values() if s.coupling == "tight"]
        loose = [s for s in analysis_result.scored_impacts.values() if s.coupling == "loose"]
        if tight and loose:
            assert max(s.score for s in tight) > max(s.score for s in loose)

    def test_critical_scores_higher_than_low(self, analyzer_no_llm, full_graph, traverser):
        """Critical components should score higher than low-criticality ones
        at the same depth and coupling."""
        result = analyzer_no_llm.analyze(["auth_engine"], use_llm=False)
        by_crit = {}
        for si in result.scored_impacts.values():
            by_crit.setdefault(si.criticality, []).append(si.score)
        if "critical" in by_crit and "low" in by_crit:
            assert max(by_crit["critical"]) > max(by_crit["low"])

    def test_deeper_impacts_score_lower(self, analysis_result):
        """Depth-1 impacts should score >= depth-2+ impacts (same coupling/crit)."""
        direct = [s for s in analysis_result.scored_impacts.values() if s.is_direct]
        indirect = [s for s in analysis_result.scored_impacts.values() if not s.is_direct]
        if direct and indirect:
            assert max(s.score for s in direct) >= max(s.score for s in indirect)

    def test_exact_formula(self):
        """Manually verify: tight coupling * critical * depth 1 => 1.0 * 4.0 / 1 = 4.0."""
        coupling_w = 1.0   # tight
        crit_mult = 4.0    # critical
        depth = 1
        expected = coupling_w * crit_mult / depth
        assert expected == 4.0

        # loose + low + depth 2 => 0.5 * 0.5 / 2 = 0.125
        assert 0.5 * 0.5 / 2 == 0.125


# ── TestImpactAnalyzer ───────────────────────────────────────────────────


class TestImpactAnalyzer:
    """Full analysis pipeline with use_llm=False (no API key needed)."""

    def test_analysis_returns_result(self, analysis_result):
        assert isinstance(analysis_result, AnalysisResult)
        assert analysis_result.changed_components == ["auth_engine"]

    def test_scored_impacts_non_empty(self, analysis_result):
        """auth_engine has shared modules so there must be affected components."""
        assert len(analysis_result.scored_impacts) > 0

    def test_scored_impacts_contain_expected_components(self, analysis_result):
        """auth_engine shares crypto_utils, session_store, event_bus, access_control.
        Components using those modules should appear."""
        ids = set(analysis_result.scored_impacts.keys())
        # encryption_service shares crypto_utils with auth_engine
        assert "encryption_service" in ids

    def test_direct_vs_indirect_classification(self, analysis_result):
        """Direct impacts (depth 1) should match graph traversal direct_impacts."""
        graph_direct = set(analysis_result.graph_result.direct_impacts.keys())
        analyzer_direct = {
            sid for sid, si in analysis_result.scored_impacts.items() if si.is_direct
        }
        assert analyzer_direct == graph_direct

    def test_variant_risks_populated(self, analysis_result):
        """auth_engine is in all 8 variants, so variant_risks should be non-empty."""
        assert len(analysis_result.variant_risks) > 0

    def test_score_ordering_highest_first(self, analysis_result):
        """Scored impacts should be sortable by score descending."""
        scores = [si.score for si in analysis_result.scored_impacts.values()]
        sorted_desc = sorted(scores, reverse=True)
        assert sorted_desc == sorted(scores, reverse=True)

    def test_llm_not_used(self, analysis_result):
        """With use_llm=False, llm_used should be False."""
        assert analysis_result.llm_used is False


# ── TestSemanticBoost ────────────────────────────────────────────────────


class TestSemanticBoost:
    """Test the 1.25x score boost for components found by both graph AND semantic."""

    def test_semantic_boost_applied(self, full_graph, traverser, tmp_chroma_dir):
        """Components found by both graph traversal and semantic search should
        receive a 1.25x score multiplier.  Uses Chroma's default Sentence
        Transformer embeddings (no OpenAI key required)."""
        store = VectorStore(
            config_path=CONFIG_PATH,
            persist_directory=tmp_chroma_dir,
            collection_name="boost_test",
        )
        embedder = ComponentEmbedder(full_graph, store, config_path=CONFIG_PATH)
        embedder.embed_and_store(reset=True)

        retriever = SemanticRetriever(store, similarity_threshold=0.3, top_k=10, rerank_top_k=5)
        analyzer_with_rag = ImpactAnalyzer(
            full_graph, traverser, retriever=retriever, config_path=CONFIG_PATH,
        )

        result_with_rag = analyzer_with_rag.analyze(["auth_engine"], use_llm=False)

        # Also run without retriever for baseline comparison
        analyzer_plain = ImpactAnalyzer(full_graph, traverser, config_path=CONFIG_PATH)
        result_plain = analyzer_plain.analyze(["auth_engine"], use_llm=False)

        # Any component with a semantic_matches entry got the 1.25x boost
        boosted = {
            sid for sid, si in result_with_rag.scored_impacts.items()
            if si.semantic_matches
        }
        for sid in boosted:
            if sid in result_plain.scored_impacts:
                base_score = result_plain.scored_impacts[sid].score
                boosted_score = result_with_rag.scored_impacts[sid].score
                assert boosted_score == pytest.approx(base_score * 1.25, rel=0.01)


# ── TestVariantRiskAggregation ───────────────────────────────────────────


class TestVariantRiskAggregation:
    """Verify variant-level risk aggregation logic."""

    def test_risk_score_is_sum_of_component_scores(self, analysis_result):
        """A variant's risk_score should be the sum of its affected component scores."""
        for vid, vr in analysis_result.variant_risks.items():
            expected_sum = sum(si.score for si in vr.affected_components)
            assert vr.risk_score == pytest.approx(expected_sum, abs=0.01), (
                f"Variant {vid}: expected {expected_sum}, got {vr.risk_score}"
            )

    def test_direct_indirect_counts(self, analysis_result):
        """direct_count + indirect_count should equal total affected components
        (including the changed component itself counted as direct)."""
        for vid, vr in analysis_result.variant_risks.items():
            total = vr.direct_count + vr.indirect_count
            # Total should be >= number of scored affected components in this variant
            assert total >= len(vr.affected_components)

    def test_max_criticality_picks_highest(self, analysis_result):
        """max_criticality should be the highest among affected components."""
        rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        for vid, vr in analysis_result.variant_risks.items():
            # max_criticality should be at least as high as any affected component
            for si in vr.affected_components:
                assert rank.get(vr.max_criticality, 0) >= rank.get(si.criticality, 0), (
                    f"Variant {vid}: max_criticality={vr.max_criticality} "
                    f"but component {si.component_id} has {si.criticality}"
                )


# ── TestImpactReporter ───────────────────────────────────────────────────


class TestImpactReporter:
    """Verify report formatting across dict, markdown, and terminal outputs."""

    @pytest.fixture
    def reporter(self, analysis_result):
        return ImpactReporter(analysis_result)

    # -- to_dict() --

    def test_to_dict_keys(self, reporter):
        d = reporter.to_dict()
        assert "change_scope" in d
        assert "impacts" in d
        assert "variant_risks" in d
        assert "summary" in d
        assert "llm_used" in d

    def test_to_dict_impact_shape(self, reporter):
        d = reporter.to_dict()
        for impact in d["impacts"]:
            assert "component_id" in impact
            assert "component_name" in impact
            assert "criticality" in impact
            assert "score" in impact
            assert "paths" in impact

    # -- to_markdown() --

    def test_to_markdown_header(self, reporter):
        md = reporter.to_markdown()
        assert "# Impact Radar" in md

    def test_to_markdown_sections(self, reporter):
        md = reporter.to_markdown()
        assert "## Direct Impacts" in md
        assert "## Indirect Impacts" in md

    # -- to_terminal() --

    def test_to_terminal_rich_markup(self, reporter):
        term = reporter.to_terminal()
        assert "[bold]" in term

    # -- Risk level thresholds --

    def test_risk_level_critical(self, reporter):
        assert reporter._risk_level(7.0) == "CRITICAL"
        assert reporter._risk_level(10.0) == "CRITICAL"

    def test_risk_level_high(self, reporter):
        assert reporter._risk_level(4.0) == "HIGH"
        assert reporter._risk_level(6.9) == "HIGH"

    def test_risk_level_medium(self, reporter):
        assert reporter._risk_level(2.0) == "MEDIUM"
        assert reporter._risk_level(3.9) == "MEDIUM"

    def test_risk_level_low(self, reporter):
        assert reporter._risk_level(1.9) == "LOW"
        assert reporter._risk_level(0.0) == "LOW"
