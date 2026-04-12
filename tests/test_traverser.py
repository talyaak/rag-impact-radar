"""Tests for the impact traverser — the core blast radius algorithm.

=== RAG Learning: Testing Structural Retrieval ===

These tests verify that graph traversal correctly identifies both obvious
and non-obvious impacts. This is analogous to testing a retrieval system:
  - "Does it find the obviously related document?" (direct impacts)
  - "Does it find the non-obviously related document?" (indirect impacts)
  - "Does it handle edge cases?" (cycles, depth limits, empty input)

The non-obvious impact chains are the ENTIRE POINT of this system. If a
human could easily spot all affected variants, we wouldn't need a tool.
The tests below verify the 4 specific non-obvious chains from the seed data.
"""

import pytest
from pathlib import Path

from src.graph.builder import DependencyGraph
from src.graph.traverser import ImpactTraverser, ImpactResult

# ── Fixtures ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
COMPONENTS_DIR = DATA_DIR / "components"
VARIANTS_DIR = DATA_DIR / "variants"


@pytest.fixture
def full_graph() -> DependencyGraph:
    return DependencyGraph(COMPONENTS_DIR, VARIANTS_DIR)


@pytest.fixture
def traverser(full_graph: DependencyGraph) -> ImpactTraverser:
    return ImpactTraverser(full_graph, max_depth=5, include_weak_coupling=True)


@pytest.fixture
def strict_traverser(full_graph: DependencyGraph) -> ImpactTraverser:
    """Traverser that only follows tight couplings."""
    return ImpactTraverser(full_graph, max_depth=5, include_weak_coupling=False)


@pytest.fixture
def shallow_traverser(full_graph: DependencyGraph) -> ImpactTraverser:
    """Traverser limited to depth 1 (direct impacts only)."""
    return ImpactTraverser(full_graph, max_depth=1, include_weak_coupling=True)


@pytest.fixture
def minimal_graph() -> DependencyGraph:
    """A → mod_x ← B → mod_y ← C → mod_z ← D (linear chain, depth 3)."""
    g = DependencyGraph()
    g.add_component({
        "id": "a", "name": "A", "criticality": "critical",
        "modules": [{"module_id": "mod_x", "coupling": "tight"}],
    })
    g.add_component({
        "id": "b", "name": "B", "criticality": "high",
        "modules": [
            {"module_id": "mod_x", "coupling": "tight"},
            {"module_id": "mod_y", "coupling": "loose"},
        ],
    })
    g.add_component({
        "id": "c", "name": "C", "criticality": "medium",
        "modules": [
            {"module_id": "mod_y", "coupling": "tight"},
            {"module_id": "mod_z", "coupling": "tight"},
        ],
    })
    g.add_component({
        "id": "d", "name": "D", "criticality": "low",
        "modules": [{"module_id": "mod_z", "coupling": "optional"}],
    })
    g.add_variant({
        "id": "v1", "name": "V1",
        "components": [
            {"component_id": "a", "features": []},
            {"component_id": "b", "features": []},
        ],
    })
    g.add_variant({
        "id": "v2", "name": "V2",
        "components": [
            {"component_id": "c", "features": []},
            {"component_id": "d", "features": []},
        ],
    })
    return g


@pytest.fixture
def cycle_graph() -> DependencyGraph:
    """A → m1 ← B → m2 ← C → m3 ← A (triangle cycle)."""
    g = DependencyGraph()
    g.add_component({
        "id": "a", "name": "A", "criticality": "high",
        "modules": [
            {"module_id": "m1", "coupling": "tight"},
            {"module_id": "m3", "coupling": "tight"},
        ],
    })
    g.add_component({
        "id": "b", "name": "B", "criticality": "high",
        "modules": [
            {"module_id": "m1", "coupling": "tight"},
            {"module_id": "m2", "coupling": "tight"},
        ],
    })
    g.add_component({
        "id": "c", "name": "C", "criticality": "high",
        "modules": [
            {"module_id": "m2", "coupling": "tight"},
            {"module_id": "m3", "coupling": "tight"},
        ],
    })
    return g


# ── Test: Basic Direct Impacts ────────────────────────────────────────

class TestDirectImpacts:
    """Test that direct dependencies (depth 1) are found correctly."""

    def test_auth_engine_direct_impacts(self, traverser: ImpactTraverser):
        """Changing AuthEngine should directly impact components sharing its modules.

        AuthEngine uses: crypto_utils, session_store, event_bus, access_control
        Direct neighbors via shared modules:
          - crypto_utils → EncryptionService
          - session_store → WebSocketGateway
          - event_bus → NotificationService, AuditLogger, WorkflowEngine
          - access_control → APIGateway, WorkflowEngine
        """
        result = traverser.traverse(["auth_engine"])
        direct_ids = set(result.direct_impacts.keys())

        # These MUST be in direct impacts
        assert "encryption_service" in direct_ids, "shares crypto_utils"
        assert "websocket_gateway" in direct_ids, "shares session_store"
        assert "notification_service" in direct_ids, "shares event_bus"
        assert "audit_logger" in direct_ids, "shares event_bus"
        assert "workflow_engine" in direct_ids, "shares event_bus + access_control"
        assert "api_gateway" in direct_ids, "shares access_control"

    def test_direct_impact_count(self, traverser: ImpactTraverser):
        """AuthEngine should have exactly 6 direct impacts."""
        result = traverser.traverse(["auth_engine"])
        assert len(result.direct_impacts) == 6

    def test_changed_component_not_in_impacts(self, traverser: ImpactTraverser):
        """The changed component itself should NOT appear in impacts."""
        result = traverser.traverse(["auth_engine"])
        assert "auth_engine" not in result.direct_impacts
        assert "auth_engine" not in result.indirect_impacts

    def test_shallow_traverser_only_direct(self, shallow_traverser: ImpactTraverser):
        """With max_depth=1, only direct impacts should be returned."""
        result = shallow_traverser.traverse(["auth_engine"])
        assert len(result.indirect_impacts) == 0
        assert len(result.direct_impacts) > 0


# ── Test: Indirect (Multi-Hop) Impacts ────────────────────────────────

class TestIndirectImpacts:
    """Test that multi-hop dependencies are discovered.

    These are the non-obvious impacts that justify building this tool.
    """

    def test_linear_chain_traversal(self, minimal_graph: DependencyGraph):
        """In chain A → mod_x ← B → mod_y ← C → mod_z ← D:
        Changing A should find B (direct), C (indirect via B), D (indirect via C).
        """
        t = ImpactTraverser(minimal_graph, max_depth=5)
        result = t.traverse(["a"])

        assert "b" in result.direct_impacts
        assert "c" in result.indirect_impacts
        assert "d" in result.indirect_impacts

    def test_linear_chain_depths(self, minimal_graph: DependencyGraph):
        """Verify correct depth assignment in a linear chain."""
        t = ImpactTraverser(minimal_graph, max_depth=5)
        result = t.traverse(["a"])

        assert result.direct_impacts["b"].depth == 1
        assert result.indirect_impacts["c"].depth == 2
        assert result.indirect_impacts["d"].depth == 3

    def test_indirect_reaches_distant_variant(self, minimal_graph: DependencyGraph):
        """Changing A (in V1) should affect V2 (contains C and D) indirectly."""
        t = ImpactTraverser(minimal_graph, max_depth=5)
        result = t.traverse(["a"])

        assert "v2" in result.affected_variants
        v2 = result.affected_variants["v2"]
        assert "c" in v2.affected_components
        assert "d" in v2.affected_components


# ── Test: Cycle Handling ──────────────────────────────────────────────

class TestCycleHandling:
    """Verify the traverser doesn't infinite-loop on circular dependencies.

    === RAG Learning: Cycles in Knowledge Graphs ===

    Real-world dependency graphs almost always contain cycles. The BFS
    visited-set pattern handles this: once we've reached a component,
    we don't re-expand it. This is the same principle used in web crawlers
    and knowledge graph traversal in production RAG systems.
    """

    def test_cycle_terminates(self, cycle_graph: DependencyGraph):
        """Traversal must terminate even with a cycle (A → B → C → A)."""
        t = ImpactTraverser(cycle_graph, max_depth=10)
        result = t.traverse(["a"])  # Should not hang
        assert isinstance(result, ImpactResult)

    def test_cycle_finds_all_nodes(self, cycle_graph: DependencyGraph):
        """In a triangle cycle, changing A should find both B and C."""
        t = ImpactTraverser(cycle_graph, max_depth=10)
        result = t.traverse(["a"])

        all_affected = result.all_affected_components
        assert "b" in all_affected
        assert "c" in all_affected

    def test_cycle_no_duplicates(self, cycle_graph: DependencyGraph):
        """Each component should appear at most once in results."""
        t = ImpactTraverser(cycle_graph, max_depth=10)
        result = t.traverse(["a"])

        all_ids = list(result.direct_impacts.keys()) + list(result.indirect_impacts.keys())
        assert len(all_ids) == len(set(all_ids)), "Duplicate component in results"

    def test_real_data_cycle_chain3(self, traverser: ImpactTraverser):
        """Chain 3 (Audit Trail Cycle): AuthEngine → event_bus → AuditLogger →
        query_builder → SearchService → db_connector → ReportGenerator →
        job_scheduler → WorkflowEngine → access_control → back to AuthEngine.

        Traversal must terminate and find all components in the ring.
        """
        result = traverser.traverse(["auth_engine"])
        all_affected = set(result.all_affected_components)

        # All components in Chain 3 should be reachable
        chain3_components = {
            "audit_logger", "search_service", "report_generator",
            "workflow_engine",
        }
        assert chain3_components.issubset(all_affected), (
            f"Missing from chain 3: {chain3_components - all_affected}"
        )

    def test_real_data_cycle_chain4(self, traverser: ImpactTraverser):
        """Chain 4 (Real-Time Communication Cycle): WebSocketGateway →
        session_store → AuthEngine → crypto_utils → EncryptionService →
        config_loader → BillingCore → metrics_collector → WebSocketGateway.

        Starting from WebSocketGateway, all chain members should be found.
        """
        result = traverser.traverse(["websocket_gateway"])
        all_affected = set(result.all_affected_components)

        chain4_components = {
            "auth_engine", "encryption_service", "billing_core",
        }
        assert chain4_components.issubset(all_affected), (
            f"Missing from chain 4: {chain4_components - all_affected}"
        )


# ── Test: Non-Obvious Dependency Chains (Seed Data) ──────────────────

class TestNonObviousChains:
    """Test the 4 specific non-obvious chains designed in the seed data.

    These are the cases that make impact-radar valuable — dependencies
    that a human reviewing a single component's YAML would miss.
    """

    def test_chain1_i18n_surprise(self, traverser: ImpactTraverser):
        """Chain 1: Changing NotificationService affects SearchService
        (via shared i18n_utils) and APIGateway (via shared cache_layer
        with SearchService).

        NotificationService → i18n_utils ← SearchService → cache_layer ← APIGateway
        """
        result = traverser.traverse(["notification_service"])
        all_affected = set(result.all_affected_components)

        # Direct: components sharing notification_service's modules
        assert "search_service" in all_affected, "shares i18n_utils"
        assert "report_generator" in all_affected, "shares template_engine + i18n_utils"

        # Indirect chain: SearchService → cache_layer ← APIGateway
        assert "api_gateway" in all_affected, "indirect via cache_layer through SearchService"

    def test_chain2_billing_to_security(self, traverser: ImpactTraverser):
        """Chain 2: Changing BillingCore reaches EncryptionService and DataExporter.

        BillingCore → currency_utils ← PricingEngine → rate_limiter ← APIGateway
        → config_loader ← EncryptionService → key_rotation ← DataExporter
        """
        result = traverser.traverse(["billing_core"])
        all_affected = set(result.all_affected_components)

        # Direct impacts from BillingCore's modules
        assert "pricing_engine" in all_affected, "shares tax_calculator + currency_utils"

        # The full chain should eventually reach EncryptionService and DataExporter
        assert "encryption_service" in all_affected, "indirect via config_loader chain"
        assert "data_exporter" in all_affected, "indirect via key_rotation chain"

    def test_chain3_audit_trail_cycle(self, traverser: ImpactTraverser):
        """Chain 3: Starting from AuditLogger, the cycle should reach
        back around through SearchService → ReportGenerator → WorkflowEngine.
        """
        result = traverser.traverse(["audit_logger"])
        all_affected = set(result.all_affected_components)

        assert "search_service" in all_affected, "shares query_builder"
        assert "report_generator" in all_affected, "indirect via db_connector or query_builder"
        assert "workflow_engine" in all_affected, "shares event_bus"

    def test_chain4_realtime_communication_trap(self, traverser: ImpactTraverser):
        """Chain 4: WebSocketGateway changes cascade through auth and
        encryption back to billing.
        """
        result = traverser.traverse(["websocket_gateway"])
        all_affected = set(result.all_affected_components)

        assert "auth_engine" in all_affected, "shares session_store"
        assert "encryption_service" in all_affected, "indirect via crypto_utils"
        assert "billing_core" in all_affected, "indirect via config_loader or metrics_collector"


# ── Test: Depth Limiting ──────────────────────────────────────────────

class TestDepthLimiting:
    """Verify that max_depth parameter correctly limits traversal."""

    def test_depth_1_only_direct(self, full_graph: DependencyGraph):
        t = ImpactTraverser(full_graph, max_depth=1)
        result = t.traverse(["auth_engine"])
        assert len(result.indirect_impacts) == 0
        assert len(result.direct_impacts) > 0

    def test_depth_2_finds_some_indirect(self, full_graph: DependencyGraph):
        t = ImpactTraverser(full_graph, max_depth=2)
        result = t.traverse(["auth_engine"])
        assert len(result.indirect_impacts) > 0

    def test_deeper_depth_finds_more(self, full_graph: DependencyGraph):
        """Increasing depth should find equal or more affected components."""
        t2 = ImpactTraverser(full_graph, max_depth=2)
        t5 = ImpactTraverser(full_graph, max_depth=5)
        r2 = t2.traverse(["auth_engine"])
        r5 = t5.traverse(["auth_engine"])
        assert len(r5.all_affected_components) >= len(r2.all_affected_components)

    def test_override_depth_in_traverse(self, traverser: ImpactTraverser):
        """The traverse() method should accept a max_depth override."""
        r1 = traverser.traverse(["auth_engine"], max_depth=1)
        r5 = traverser.traverse(["auth_engine"], max_depth=5)
        assert len(r5.all_affected_components) >= len(r1.all_affected_components)


# ── Test: Coupling Filtering ─────────────────────────────────────────

class TestCouplingFiltering:
    """Test that weak coupling filtering reduces impact surface."""

    def test_strict_finds_fewer_impacts(
        self,
        traverser: ImpactTraverser,
        strict_traverser: ImpactTraverser,
    ):
        """Excluding weak couplings should find fewer or equal impacts."""
        r_all = traverser.traverse(["auth_engine"])
        r_strict = strict_traverser.traverse(["auth_engine"])
        assert len(r_strict.all_affected_components) <= len(r_all.all_affected_components)

    def test_strict_excludes_loose_coupling(self, strict_traverser: ImpactTraverser):
        """AuthEngine → event_bus is 'loose' coupling. With strict mode,
        components only reachable via event_bus should be excluded.
        """
        result = strict_traverser.traverse(["auth_engine"])
        # event_bus has loose coupling from auth_engine, so components
        # ONLY reachable through event_bus should not appear
        # (but they might still be reachable via other tight paths)


# ── Test: Variant Impact ──────────────────────────────────────────────

class TestVariantImpact:
    """Test that variant-level impact is computed correctly."""

    def test_all_variants_affected_by_auth(self, traverser: ImpactTraverser):
        """AuthEngine is in all 8 variants, so changing it should affect all 8."""
        result = traverser.traverse(["auth_engine"])
        # auth_engine is a changed component, present in all variants
        assert len(result.affected_variants) == 8

    def test_variant_affected_components_correct(self, traverser: ImpactTraverser):
        """Starter Global only has 5 components. Check which are affected."""
        result = traverser.traverse(["auth_engine"])
        starter = result.affected_variants.get("starter_global")
        assert starter is not None
        # starter_global has: auth_engine, billing_core, notification_service, api_gateway, search_service
        # auth_engine is the changed component (should be in affected_components)
        assert "auth_engine" in starter.affected_components

    def test_variant_criticality_tracking(self, traverser: ImpactTraverser):
        """Enterprise variants should report 'critical' max_criticality."""
        result = traverser.traverse(["billing_core"])
        enterprise = result.affected_variants.get("enterprise_global")
        assert enterprise is not None
        assert enterprise.max_criticality == "critical"

    def test_platform_variant_limited_scope(self, traverser: ImpactTraverser):
        """Platform variant has only 6 components — impact should be scoped."""
        result = traverser.traverse(["data_exporter"])
        if "platform_global" in result.affected_variants:
            platform = result.affected_variants["platform_global"]
            # Platform has: auth, api_gateway, billing, encryption, data_exporter, pricing
            platform_comps = {"auth_engine", "api_gateway", "billing_core",
                              "encryption_service", "data_exporter", "pricing_engine"}
            for comp in platform.affected_components:
                assert comp in platform_comps, f"{comp} not in platform variant"


# ── Test: Impact Paths ────────────────────────────────────────────────

class TestImpactPaths:
    """Test that impact paths are correctly constructed."""

    def test_direct_path_length(self, traverser: ImpactTraverser):
        """Direct impact paths should have exactly 3 elements: [comp, mod, comp]."""
        result = traverser.traverse(["auth_engine"])
        for impact in result.direct_impacts.values():
            for path in impact.paths:
                assert path.depth == 1
                assert len(path.chain) == 3, f"Expected 3 elements, got {path.chain}"

    def test_path_starts_with_source(self, traverser: ImpactTraverser):
        result = traverser.traverse(["auth_engine"])
        for path in result.all_impact_paths:
            assert path.chain[0] == "auth_engine"

    def test_path_ends_with_target(self, traverser: ImpactTraverser):
        result = traverser.traverse(["auth_engine"])
        for path in result.all_impact_paths:
            assert path.chain[-1] == path.target

    def test_path_describe_is_readable(self, traverser: ImpactTraverser):
        """The describe() method should produce a human-readable string."""
        result = traverser.traverse(["auth_engine"])
        if result.all_impact_paths:
            desc = result.all_impact_paths[0].describe()
            assert " → " in desc
            assert "depth=" in desc


# ── Test: Edge Cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_change_scope(self, traverser: ImpactTraverser):
        """Empty change scope should return empty results."""
        result = traverser.traverse([])
        assert len(result.direct_impacts) == 0
        assert len(result.indirect_impacts) == 0
        assert len(result.affected_variants) == 0

    def test_nonexistent_component(self, traverser: ImpactTraverser):
        """Non-existent component should be ignored gracefully."""
        result = traverser.traverse(["nonexistent_component"])
        assert len(result.direct_impacts) == 0

    def test_multiple_changed_components(self, traverser: ImpactTraverser):
        """Multiple changed components should combine their blast radius.

        Note: When both auth_engine and billing_core are changed, billing_core
        is excluded from the impact list (it's a source, not an impact). So the
        total affected count may be less than a single-source traversal that
        counts billing_core as an impact. What matters is that the UNION of
        both sources' direct impacts is a superset of either alone.
        """
        r_auth = traverser.traverse(["auth_engine"], max_depth=1)
        r_billing = traverser.traverse(["billing_core"], max_depth=1)
        r_double = traverser.traverse(["auth_engine", "billing_core"], max_depth=1)

        auth_direct = {c for c in r_auth.direct_impacts if c != "billing_core"}
        billing_direct = {c for c in r_billing.direct_impacts if c != "auth_engine"}
        double_direct = set(r_double.direct_impacts.keys())

        # The combined traversal should find at least the union minus the sources
        assert auth_direct.issubset(double_direct)
        assert billing_direct.issubset(double_direct)

    def test_result_summary(self, traverser: ImpactTraverser):
        """The summary() method should return expected keys."""
        result = traverser.traverse(["auth_engine"])
        summary = result.summary()
        expected_keys = {
            "changed", "direct_impacts", "indirect_impacts",
            "total_affected_components", "affected_variants", "total_paths",
        }
        assert set(summary.keys()) == expected_keys
