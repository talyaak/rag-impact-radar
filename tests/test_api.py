"""Tests for the FastAPI REST layer — health, CRUD, graph, and analysis endpoints.

=== RAG Pipeline Learning: Testing a RAG Serving Layer ===

The API is the external interface to the entire RAG pipeline. Testing it
verifies that:
  1. The lifespan initialises the dependency graph and vector store correctly.
  2. Component and variant CRUD endpoints expose the knowledge graph faithfully.
  3. The /analyze endpoint orchestrates graph traversal + semantic retrieval
     and returns a well-structured response.
  4. Error handling works: unknown components return 400, missing bodies 422.

Key testing strategy:
  - Use FastAPI's ``TestClient`` as a context manager so the app lifespan
    runs and the dependency graph is loaded before any requests.
  - Always set ``use_llm: false`` in analyze requests to avoid needing an
    OpenAI API key.
  - Assert on response shapes rather than exact values where the seed data
    might evolve — but pin counts (12 components, 8 variants) to the current
    seed data for regression detection.
===
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

# ── Shared client (context-managed so the lifespan runs) ─────────────────


@pytest.fixture(scope="module")
def client():
    """Yield a TestClient whose lifespan has initialised the graph."""
    with TestClient(app) as c:
        yield c


# ── TestHealthEndpoint ───────────────────────────────────────────────────


class TestHealthEndpoint:
    """The /health probe should always return 200 with status ok."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        resp = client.get("/health")
        assert resp.json() == {"status": "ok"}


# ── TestComponentEndpoints ───────────────────────────────────────────────


class TestComponentEndpoints:
    """Verify component listing and lookup against the seed data."""

    def test_list_components_count(self, client):
        """There are exactly 12 components in the seed data."""
        resp = client.get("/api/v1/components")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 12

    def test_list_components_shape(self, client):
        """Each component dict should contain the expected keys."""
        resp = client.get("/api/v1/components")
        for comp in resp.json():
            assert "id" in comp
            assert "name" in comp
            assert "description" in comp
            assert "team_owner" in comp
            assert "criticality" in comp

    def test_get_known_component(self, client):
        """GET /api/v1/components/auth_engine returns the correct component."""
        resp = client.get("/api/v1/components/auth_engine")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "auth_engine"
        assert data["name"] == "AuthEngine"
        assert data["criticality"] == "critical"

    def test_get_nonexistent_component(self, client):
        """GET /api/v1/components/nonexistent returns 404."""
        resp = client.get("/api/v1/components/nonexistent")
        assert resp.status_code == 404


# ── TestVariantEndpoints ─────────────────────────────────────────────────


class TestVariantEndpoints:
    """Verify variant listing and lookup against the seed data."""

    def test_list_variants_count(self, client):
        """There are exactly 8 variants in the seed data."""
        resp = client.get("/api/v1/variants")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 8

    def test_list_variants_shape(self, client):
        """Each variant dict should contain the expected keys."""
        resp = client.get("/api/v1/variants")
        for v in resp.json():
            assert "id" in v
            assert "name" in v
            assert "tier" in v
            assert "region" in v

    def test_get_known_variant(self, client):
        """GET /api/v1/variants/starter_global returns the correct variant."""
        resp = client.get("/api/v1/variants/starter_global")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "starter_global"
        assert data["name"] == "Starter"
        assert data["tier"] == "starter"
        assert data["region"] == "global"

    def test_get_nonexistent_variant(self, client):
        """GET /api/v1/variants/nonexistent returns 404."""
        resp = client.get("/api/v1/variants/nonexistent")
        assert resp.status_code == 404


# ── TestGraphEndpoints ───────────────────────────────────────────────────


class TestGraphEndpoints:
    """Verify graph summary statistics."""

    def test_graph_summary_status(self, client):
        resp = client.get("/api/v1/graph/summary")
        assert resp.status_code == 200

    def test_graph_summary_counts(self, client):
        """component_count == 12 and variant_count == 8 in the seed data."""
        resp = client.get("/api/v1/graph/summary")
        data = resp.json()
        assert data["component_count"] == 12
        assert data["variant_count"] == 8

    def test_graph_summary_keys(self, client):
        resp = client.get("/api/v1/graph/summary")
        data = resp.json()
        for key in ("total_nodes", "component_count", "module_count",
                     "edge_count", "variant_count", "shared_module_count"):
            assert key in data


# ── TestAnalyzeEndpoint ──────────────────────────────────────────────────


class TestAnalyzeEndpoint:
    """Verify the /analyze endpoint with use_llm=false (no API key needed)."""

    def test_analyze_auth_engine(self, client):
        """POST /api/v1/analyze with auth_engine returns a valid response."""
        resp = client.post(
            "/api/v1/analyze",
            json={"changed_components": ["auth_engine"], "use_llm": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "changed_components" in data
        assert "auth_engine" in data["changed_components"]

    def test_analyze_response_shape(self, client):
        """The response should contain the expected top-level keys."""
        resp = client.post(
            "/api/v1/analyze",
            json={"changed_components": ["auth_engine"], "use_llm": False},
        )
        data = resp.json()
        for key in ("changed_components", "direct_impacts", "indirect_impacts",
                     "affected_variants", "summary", "llm_used"):
            assert key in data, f"Missing key: {key}"

    def test_analyze_has_direct_impacts(self, client):
        """auth_engine shares modules with several components -- expect direct hits."""
        resp = client.post(
            "/api/v1/analyze",
            json={"changed_components": ["auth_engine"], "use_llm": False},
        )
        data = resp.json()
        assert len(data["direct_impacts"]) > 0

    def test_analyze_has_affected_variants(self, client):
        """auth_engine is in all 8 variants, so affected_variants should be non-empty."""
        resp = client.post(
            "/api/v1/analyze",
            json={"changed_components": ["auth_engine"], "use_llm": False},
        )
        data = resp.json()
        assert len(data["affected_variants"]) > 0

    def test_analyze_llm_not_used(self, client):
        """With use_llm=false, the response should report llm_used=false."""
        resp = client.post(
            "/api/v1/analyze",
            json={"changed_components": ["auth_engine"], "use_llm": False},
        )
        data = resp.json()
        assert data["llm_used"] is False

    def test_analyze_unknown_component_returns_400(self, client):
        """Posting an unknown component ID should return 400."""
        resp = client.post(
            "/api/v1/analyze",
            json={"changed_components": ["does_not_exist"], "use_llm": False},
        )
        assert resp.status_code == 400

    def test_analyze_empty_list_returns_422(self, client):
        """An empty changed_components list violates min_length=1 validation."""
        resp = client.post(
            "/api/v1/analyze",
            json={"changed_components": [], "use_llm": False},
        )
        assert resp.status_code == 422

    def test_analyze_multiple_components(self, client):
        """Analyze two components at once -- should succeed."""
        resp = client.post(
            "/api/v1/analyze",
            json={
                "changed_components": ["auth_engine", "billing_core"],
                "use_llm": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert set(data["changed_components"]) == {"auth_engine", "billing_core"}

    def test_analyze_summary_counts(self, client):
        """Summary should report non-negative impact counts."""
        resp = client.post(
            "/api/v1/analyze",
            json={"changed_components": ["auth_engine"], "use_llm": False},
        )
        summary = resp.json()["summary"]
        assert summary["direct_impacts"] >= 0
        assert summary["indirect_impacts"] >= 0
        assert summary["affected_variants"] >= 0
