"""Tests for the V2 FastAPI endpoints — ingestion, gap analysis, and recompilation.

Tests verify:
  - V2 status endpoint returns correct version and state
  - YAML ingestion endpoint loads components and variants
  - Gap detection and session management work via API
  - Test generation endpoint produces valid output
  - All tests run without API keys (zero-API-key testing)
"""

import textwrap
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


# ── Fixtures ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
COMPONENTS_DIR = DATA_DIR / "components"
VARIANTS_DIR = DATA_DIR / "variants"


@pytest.fixture
def client():
    """Create a test client with the app lifecycle."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a minimal sample repository."""
    svc = tmp_path / "myservice"
    svc.mkdir()
    (svc / "__init__.py").write_text('"""My service."""\n')
    (svc / "handler.py").write_text(textwrap.dedent('''\
        """Request handler."""

        class RequestHandler:
            """Handles incoming requests."""

            def get(self, path: str) -> dict:
                return {"path": path}

            def post(self, path: str, body: dict) -> dict:
                return {"path": path, "body": body}

            def delete(self, path: str) -> bool:
                return True
    '''))
    return tmp_path


# ── V2 Status Tests ──────────────────────────────────────────────────


class TestV2Status:
    def test_v2_status_returns_version(self, client):
        resp = client.get("/api/v2/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "2.0.0"
        assert "graph_loaded" in data

    def test_health_still_works(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── V2 Ingestion Tests ───────────────────────────────────────────────


class TestV2Ingestion:
    def test_ingest_from_yaml(self, client):
        resp = client.post("/api/v2/ingest/yaml", json={
            "components_dir": str(COMPONENTS_DIR),
            "variants_dir": str(VARIANTS_DIR),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["components_ingested"] == 12

    def test_ingest_codebase(self, client, sample_repo):
        resp = client.post("/api/v2/ingest", json={
            "repo_path": str(sample_repo),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "scan" in data
        assert "parse" in data
        assert "graph" in data

    def test_ingest_updates_graph(self, client):
        """After ingestion, graph summary should reflect loaded data."""
        client.post("/api/v2/ingest/yaml", json={
            "components_dir": str(COMPONENTS_DIR),
            "variants_dir": str(VARIANTS_DIR),
        })

        resp = client.get("/api/v1/graph/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["component_count"] == 12


# ── V2 Gap Analysis Tests ────────────────────────────────────────────


class TestV2GapAnalysis:
    def test_detect_gaps(self, client):
        # First load data
        client.post("/api/v2/ingest/yaml", json={
            "components_dir": str(COMPONENTS_DIR),
            "variants_dir": str(VARIANTS_DIR),
        })

        resp = client.post("/api/v2/gaps/detect")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_gaps" in data
        assert "completeness_score" in data

    def test_start_gap_session(self, client):
        client.post("/api/v2/ingest/yaml", json={
            "components_dir": str(COMPONENTS_DIR),
            "variants_dir": str(VARIANTS_DIR),
        })

        resp = client.post("/api/v2/gaps/start-session")
        assert resp.status_code == 200
        data = resp.json()
        assert "session" in data
        assert "questions" in data

    def test_get_gap_questions(self, client):
        client.post("/api/v2/ingest/yaml", json={
            "components_dir": str(COMPONENTS_DIR),
            "variants_dir": str(VARIANTS_DIR),
        })
        client.post("/api/v2/gaps/start-session")

        resp = client.get("/api/v2/gaps/questions")
        assert resp.status_code == 200
        data = resp.json()
        assert "questions" in data

    def test_get_questions_without_session_returns_404(self, client):
        import src.api.main as api_mod
        api_mod._gap_session = None  # Reset session state from prior tests
        resp = client.get("/api/v2/gaps/questions")
        assert resp.status_code == 404

    def test_answer_questions(self, client):
        client.post("/api/v2/ingest/yaml", json={
            "components_dir": str(COMPONENTS_DIR),
            "variants_dir": str(VARIANTS_DIR),
        })
        session_resp = client.post("/api/v2/gaps/start-session")
        questions = session_resp.json().get("questions", [])

        if not questions:
            pytest.skip("No questions to answer")

        resp = client.post("/api/v2/gaps/answer", json={
            "answers": [{
                "gap_index": questions[0]["gap_index"],
                "answer": "This is a test answer",
                "accept_suggestion": False,
            }],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "session" in data
        assert data["session"]["iteration"] > 0


# ── V2 Test Generation Tests ─────────────────────────────────────────


class TestV2TestGeneration:
    def test_generate_tests(self, client):
        client.post("/api/v2/ingest/yaml", json={
            "components_dir": str(COMPONENTS_DIR),
            "variants_dir": str(VARIANTS_DIR),
        })

        resp = client.post("/api/v2/generate-tests")
        assert resp.status_code == 200
        data = resp.json()
        assert "tests_generated" in data
        assert data["tests_generated"] > 0
        assert "test_file" in data


# ── V1 Backward Compatibility Tests ──────────────────────────────────


class TestV1Compatibility:
    def test_v1_analyze_still_works(self, client):
        """V1 analyze endpoint should still function after V2 changes."""
        # Load data via V1 path (lifespan loads seed data)
        resp = client.post("/api/v1/analyze", json={
            "changed_components": ["auth_engine"],
            "use_llm": False,
            "max_depth": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "changed_components" in data
        assert "direct_impacts" in data

    def test_v1_components_endpoint(self, client):
        resp = client.get("/api/v1/components")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 12

    def test_v1_variants_endpoint(self, client):
        resp = client.get("/api/v1/variants")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 8
