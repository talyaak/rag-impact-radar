"""Tests for the dependency graph builder.

=== RAG Learning: Why Test the Graph? ===

The graph is the foundation of the "structured retrieval" half of our
hybrid RAG pipeline. If the graph is wrong, traversal produces wrong
impact paths, the LLM gets wrong evidence, and the risk report is wrong.
Garbage in, garbage out — at every layer.

These tests verify:
  1. YAML loading produces correct node and edge counts
  2. Bipartite structure is maintained (edges only cross node types)
  3. Shared module detection works (the key insight for impact analysis)
  4. Edge metadata (coupling, usage) is preserved (needed for grounded generation)
  5. Variant-to-component mapping is correct (needed for blast radius)
"""

import pytest
from pathlib import Path

from src.graph.builder import DependencyGraph

# ── Fixtures ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
COMPONENTS_DIR = DATA_DIR / "components"
VARIANTS_DIR = DATA_DIR / "variants"


@pytest.fixture
def full_graph() -> DependencyGraph:
    """Load the complete graph from seed data."""
    return DependencyGraph(COMPONENTS_DIR, VARIANTS_DIR)


@pytest.fixture
def minimal_graph() -> DependencyGraph:
    """Build a small graph programmatically for isolated tests.

    This creates a tiny graph:
      comp_a → mod_x ← comp_b
      comp_a → mod_y
      comp_c → mod_y

    So comp_a and comp_b share mod_x, comp_a and comp_c share mod_y.
    """
    g = DependencyGraph()
    g.add_component({
        "id": "comp_a",
        "name": "CompA",
        "description": "Component A",
        "team_owner": "team-alpha",
        "criticality": "high",
        "modules": [
            {"module_id": "mod_x", "usage": "Does X things", "coupling": "tight"},
            {"module_id": "mod_y", "usage": "Does Y things", "coupling": "loose"},
        ],
    })
    g.add_component({
        "id": "comp_b",
        "name": "CompB",
        "description": "Component B",
        "team_owner": "team-beta",
        "criticality": "medium",
        "modules": [
            {"module_id": "mod_x", "usage": "Also does X things", "coupling": "tight"},
        ],
    })
    g.add_component({
        "id": "comp_c",
        "name": "CompC",
        "description": "Component C",
        "team_owner": "team-gamma",
        "criticality": "low",
        "modules": [
            {"module_id": "mod_y", "usage": "Needs Y things", "coupling": "optional"},
        ],
    })
    g.add_variant({
        "id": "variant_1",
        "name": "Variant 1",
        "tier": "starter",
        "region": "global",
        "description": "Test variant with A and B",
        "max_seats": 10,
        "components": [
            {"component_id": "comp_a", "features": ["f1"], "config_overrides": {}},
            {"component_id": "comp_b", "features": ["f2"], "config_overrides": {}},
        ],
        "metadata": {"launched": "2024-01-01", "compliance": [], "sla_tier": "standard"},
    })
    g.add_variant({
        "id": "variant_2",
        "name": "Variant 2",
        "tier": "professional",
        "region": "global",
        "description": "Test variant with all three",
        "max_seats": 100,
        "components": [
            {"component_id": "comp_a", "features": ["f1"], "config_overrides": {}},
            {"component_id": "comp_b", "features": ["f2"], "config_overrides": {}},
            {"component_id": "comp_c", "features": ["f3"], "config_overrides": {}},
        ],
        "metadata": {"launched": "2024-01-01", "compliance": [], "sla_tier": "premium"},
    })
    return g


# ── Test: Full Graph Loading ──────────────────────────────────────────

class TestFullGraphLoading:
    """Tests that verify the real seed data loads correctly."""

    def test_component_count(self, full_graph: DependencyGraph):
        """We defined exactly 12 components in the seed data."""
        assert len(full_graph.get_components()) == 12

    def test_variant_count(self, full_graph: DependencyGraph):
        """We defined exactly 8 variants in the seed data."""
        assert len(full_graph.get_variants()) == 8

    def test_module_count(self, full_graph: DependencyGraph):
        """We defined exactly 20 shared modules across components."""
        summary = full_graph.summary()
        assert summary["module_count"] == 20

    def test_all_modules_shared(self, full_graph: DependencyGraph):
        """Every module should be used by at least 2 components."""
        shared = full_graph.get_shared_modules()
        # All 20 modules should appear as shared
        assert len(shared) == 20
        for module_id, components in shared.items():
            assert len(components) >= 2, f"{module_id} used by only {len(components)} components"

    def test_total_node_count(self, full_graph: DependencyGraph):
        """Total nodes = 12 components + 20 modules = 32."""
        summary = full_graph.summary()
        assert summary["total_nodes"] == 32

    def test_known_component_exists(self, full_graph: DependencyGraph):
        """Spot-check that a specific component loaded correctly."""
        auth = full_graph.get_component("auth_engine")
        assert auth is not None
        assert auth.name == "AuthEngine"
        assert auth.criticality == "critical"
        assert auth.team_owner == "identity-team"

    def test_known_variant_exists(self, full_graph: DependencyGraph):
        """Spot-check that a specific variant loaded correctly."""
        starter = full_graph.get_variant("starter_global")
        assert starter is not None
        assert starter.name == "Starter"
        assert starter.tier == "starter"
        assert starter.region == "global"
        assert len(starter.components) == 5


# ── Test: Bipartite Structure ─────────────────────────────────────────

class TestBipartiteStructure:
    """Verify the graph maintains proper bipartite structure.

    In a bipartite graph, edges ONLY connect nodes of different types.
    Component-to-component and module-to-module edges would be a bug.
    """

    def test_no_component_to_component_edges(self, full_graph: DependencyGraph):
        graph = full_graph.graph
        for u, v in graph.edges():
            u_type = graph.nodes[u].get("node_type")
            v_type = graph.nodes[v].get("node_type")
            assert u_type != v_type, f"Same-type edge found: {u} ({u_type}) — {v} ({v_type})"

    def test_all_nodes_have_type(self, full_graph: DependencyGraph):
        """Every node should have a node_type attribute."""
        for node, data in full_graph.graph.nodes(data=True):
            assert "node_type" in data, f"Node {node} missing node_type"
            assert data["node_type"] in ("component", "module"), f"Invalid node_type: {data['node_type']}"


# ── Test: Edge Metadata ───────────────────────────────────────────────

class TestEdgeMetadata:
    """Verify that coupling and usage metadata is preserved on edges.

    This metadata is critical for the RAG pipeline — it provides the
    factual grounding that prevents the LLM from hallucinating about
    WHY two components are related.
    """

    def test_coupling_preserved(self, full_graph: DependencyGraph):
        """AuthEngine → event_bus should have 'loose' coupling."""
        coupling = full_graph.get_edge_coupling("auth_engine", "event_bus")
        assert coupling == "loose"

    def test_tight_coupling_preserved(self, full_graph: DependencyGraph):
        """AuthEngine → crypto_utils should have 'tight' coupling."""
        coupling = full_graph.get_edge_coupling("auth_engine", "crypto_utils")
        assert coupling == "tight"

    def test_usage_preserved(self, full_graph: DependencyGraph):
        """Edge usage descriptions should be non-empty strings."""
        usage = full_graph.get_edge_usage("auth_engine", "crypto_utils")
        assert usage is not None
        assert "hashing" in usage.lower() or "jwt" in usage.lower()

    def test_nonexistent_edge_returns_none(self, full_graph: DependencyGraph):
        """Querying a non-existent edge should return None, not crash."""
        assert full_graph.get_edge_coupling("auth_engine", "tax_calculator") is None


# ── Test: Component-Module Queries ────────────────────────────────────

class TestComponentModuleQueries:
    """Test the query methods that bridge components and modules."""

    def test_get_component_modules(self, full_graph: DependencyGraph):
        """AuthEngine should have exactly 4 modules."""
        modules = full_graph.get_component_modules("auth_engine")
        module_ids = {m["module_id"] for m in modules}
        assert module_ids == {"crypto_utils", "session_store", "event_bus", "access_control"}

    def test_get_module_components(self, full_graph: DependencyGraph):
        """event_bus should be shared by 4 components."""
        components = full_graph.get_module_components("event_bus")
        assert set(components) == {"auth_engine", "notification_service", "audit_logger", "workflow_engine"}

    def test_nonexistent_component_returns_empty(self, full_graph: DependencyGraph):
        """Querying a nonexistent component should return an empty list."""
        assert full_graph.get_component_modules("nonexistent") == []

    def test_nonexistent_module_returns_empty(self, full_graph: DependencyGraph):
        assert full_graph.get_module_components("nonexistent") == []


# ── Test: Variant Mapping ─────────────────────────────────────────────

class TestVariantMapping:
    """Test variant-to-component reverse lookups."""

    def test_auth_engine_in_all_variants(self, full_graph: DependencyGraph):
        """auth_engine appears in all 8 variants."""
        variants = full_graph.get_variants_for_component("auth_engine")
        assert len(variants) == 8

    def test_data_exporter_in_subset(self, full_graph: DependencyGraph):
        """data_exporter only appears in enterprise + platform variants."""
        variants = full_graph.get_variants_for_component("data_exporter")
        variant_ids = set(variants)
        # Should be in: enterprise_global, enterprise_eu, enterprise_apac, platform_global
        assert "enterprise_global" in variant_ids
        assert "platform_global" in variant_ids
        assert "starter_global" not in variant_ids

    def test_nonexistent_component_returns_empty_variants(self, full_graph: DependencyGraph):
        assert full_graph.get_variants_for_component("nonexistent") == []


# ── Test: Minimal Graph (Programmatic) ────────────────────────────────

class TestMinimalGraph:
    """Tests using a small programmatically-built graph for isolation."""

    def test_node_counts(self, minimal_graph: DependencyGraph):
        summary = minimal_graph.summary()
        assert summary["component_count"] == 3  # comp_a, comp_b, comp_c
        assert summary["module_count"] == 2  # mod_x, mod_y

    def test_shared_modules(self, minimal_graph: DependencyGraph):
        shared = minimal_graph.get_shared_modules()
        assert "mod_x" in shared
        assert "mod_y" in shared
        assert set(shared["mod_x"]) == {"comp_a", "comp_b"}
        assert set(shared["mod_y"]) == {"comp_a", "comp_c"}

    def test_variant_mapping(self, minimal_graph: DependencyGraph):
        v1_comps = minimal_graph.get_variants_for_component("comp_a")
        assert set(v1_comps) == {"variant_1", "variant_2"}

        v_c = minimal_graph.get_variants_for_component("comp_c")
        assert v_c == ["variant_2"]

    def test_edge_coupling_types(self, minimal_graph: DependencyGraph):
        assert minimal_graph.get_edge_coupling("comp_a", "mod_x") == "tight"
        assert minimal_graph.get_edge_coupling("comp_a", "mod_y") == "loose"
        assert minimal_graph.get_edge_coupling("comp_c", "mod_y") == "optional"


# ── Test: Summary ─────────────────────────────────────────────────────

class TestSummary:
    def test_summary_keys(self, full_graph: DependencyGraph):
        summary = full_graph.summary()
        expected_keys = {
            "total_nodes",
            "component_count",
            "module_count",
            "edge_count",
            "variant_count",
            "shared_module_count",
        }
        assert set(summary.keys()) == expected_keys

    def test_edge_count_reasonable(self, full_graph: DependencyGraph):
        """Total edges should equal sum of all modules across all components.

        We have 12 components with varying module counts summing to ~57 edges.
        """
        summary = full_graph.summary()
        # Each component-module pair = 1 edge, total should be > 40
        assert summary["edge_count"] > 40
        assert summary["edge_count"] < 100  # sanity upper bound
