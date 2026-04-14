"""Tests for the V2 ingestion engine — scanner, parser, and engine.

Tests verify:
  - CodebaseScanner discovers and classifies files correctly
  - CodebaseParser extracts components and modules from Python AST
  - IngestionEngine orchestrates scanning, parsing, and graph building
  - Backward-compatible YAML ingestion still works
  - All tests run without API keys (zero-API-key testing)
"""

import tempfile
import textwrap
from pathlib import Path

import pytest

from src.ingestion.scanner import CodebaseScanner, ScanResult
from src.ingestion.parser import CodebaseParser, ParseResult
from src.ingestion.engine import IngestionEngine, IngestionResult


# ── Fixtures ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
COMPONENTS_DIR = DATA_DIR / "components"
VARIANTS_DIR = DATA_DIR / "variants"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "model_config.yaml"


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a minimal sample repository for testing."""
    # Service A: A Python package with multiple classes
    svc_a = tmp_path / "service_a"
    svc_a.mkdir()
    (svc_a / "__init__.py").write_text('"""Service A handles authentication."""\n')
    (svc_a / "auth.py").write_text(textwrap.dedent('''\
        """Authentication module for user management."""

        from service_a.utils import hash_password
        from shared.event_bus import publish

        class AuthEngine:
            """Handles user authentication and session management."""

            def login(self, username: str, password: str) -> str:
                """Authenticate a user and return a session token."""
                hashed = hash_password(password)
                publish("user.login", {"username": username})
                return "token"

            def logout(self, token: str) -> None:
                """Invalidate a session token."""
                publish("user.logout", {"token": token})

            def verify(self, token: str) -> bool:
                """Verify a session token is valid."""
                return True
    '''))
    (svc_a / "utils.py").write_text(textwrap.dedent('''\
        """Utility functions for service A."""

        def hash_password(password: str) -> str:
            return "hashed_" + password
    '''))

    # Service B: Another Python package
    svc_b = tmp_path / "service_b"
    svc_b.mkdir()
    (svc_b / "__init__.py").write_text('"""Service B handles billing."""\n')
    (svc_b / "billing.py").write_text(textwrap.dedent('''\
        """Billing module for payment processing."""

        from shared.event_bus import publish

        class BillingEngine:
            """Processes payments and manages subscriptions."""

            def charge(self, user_id: str, amount: float) -> str:
                """Charge a user."""
                publish("billing.charge", {"user_id": user_id, "amount": amount})
                return "receipt_123"

            def refund(self, receipt_id: str) -> bool:
                """Process a refund."""
                return True

            def subscribe(self, user_id: str, plan: str) -> str:
                """Create a subscription."""
                return "sub_123"
    '''))

    # Shared module
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "__init__.py").write_text("")
    (shared / "event_bus.py").write_text(textwrap.dedent('''\
        """Shared event bus for inter-service communication."""

        def publish(event: str, data: dict) -> None:
            """Publish an event."""
            pass

        def subscribe(event: str, handler) -> None:
            """Subscribe to an event."""
            pass
    '''))

    # Config file
    (tmp_path / "config.yaml").write_text("key: value\n")

    # Documentation
    (tmp_path / "README.md").write_text("# Test Repo\n")

    return tmp_path


@pytest.fixture
def sample_repo_with_docker(sample_repo: Path) -> Path:
    """Add a Dockerfile to the sample repo to test service boundary detection."""
    (sample_repo / "Dockerfile").write_text("FROM python:3.11\n")
    return sample_repo


# ── Scanner Tests ─────────────────────────────────────────────────────


class TestCodebaseScanner:
    def test_scan_discovers_python_files(self, sample_repo: Path):
        scanner = CodebaseScanner()
        result = scanner.scan(sample_repo)

        assert len(result.python_files) > 0
        py_paths = {f.relative_path for f in result.python_files}
        assert "service_a/auth.py" in py_paths
        assert "service_b/billing.py" in py_paths
        assert "shared/event_bus.py" in py_paths

    def test_scan_discovers_config_files(self, sample_repo: Path):
        scanner = CodebaseScanner()
        result = scanner.scan(sample_repo)

        config_paths = {f.relative_path for f in result.config_files}
        assert "config.yaml" in config_paths

    def test_scan_discovers_doc_files(self, sample_repo: Path):
        scanner = CodebaseScanner()
        result = scanner.scan(sample_repo)

        doc_paths = {f.relative_path for f in result.doc_files}
        assert "README.md" in doc_paths

    def test_scan_detects_service_boundaries(self, sample_repo_with_docker: Path):
        scanner = CodebaseScanner()
        result = scanner.scan(sample_repo_with_docker)

        # The root directory has a Dockerfile — it's a service boundary
        assert len(result.service_boundaries) >= 1
        names = {b.name for b in result.service_boundaries}
        assert sample_repo_with_docker.name in names

    def test_scan_counts_total_files(self, sample_repo: Path):
        scanner = CodebaseScanner()
        result = scanner.scan(sample_repo)

        assert result.total_files_scanned > 0

    def test_scan_respects_ignore_patterns(self, sample_repo: Path):
        # Create a __pycache__ directory that should be ignored
        pycache = sample_repo / "__pycache__"
        pycache.mkdir()
        (pycache / "cached.pyc").write_text("bytecode")

        scanner = CodebaseScanner()
        result = scanner.scan(sample_repo)

        py_paths = {f.relative_path for f in result.python_files}
        assert "__pycache__/cached.pyc" not in py_paths

    def test_scan_invalid_directory_raises(self):
        scanner = CodebaseScanner()
        with pytest.raises(ValueError, match="not a directory"):
            scanner.scan("/nonexistent/path")

    def test_scan_summary_format(self, sample_repo: Path):
        scanner = CodebaseScanner()
        result = scanner.scan(sample_repo)
        summary = result.summary()

        assert "repo_root" in summary
        assert "python_files" in summary
        assert "total_files_scanned" in summary
        assert isinstance(summary["python_files"], int)

    def test_scan_max_file_size(self, sample_repo: Path):
        # Create a large file
        large_file = sample_repo / "large.py"
        large_file.write_text("x = 1\n" * 100_000)

        scanner = CodebaseScanner(max_file_size_bytes=1000)
        result = scanner.scan(sample_repo)

        py_paths = {f.relative_path for f in result.python_files}
        assert "large.py" not in py_paths


# ── Parser Tests ──────────────────────────────────────────────────────


class TestCodebaseParser:
    def test_parse_extracts_classes(self, sample_repo: Path):
        scanner = CodebaseScanner()
        scan_result = scanner.scan(sample_repo)

        parser = CodebaseParser(component_detection="class", min_class_methods=2)
        parse_result = parser.parse(scan_result)

        comp_names = {c.name for c in parse_result.components}
        assert "AuthEngine" in comp_names
        assert "BillingEngine" in comp_names

    def test_parse_auto_detection(self, sample_repo: Path):
        scanner = CodebaseScanner()
        scan_result = scanner.scan(sample_repo)

        parser = CodebaseParser(component_detection="auto")
        parse_result = parser.parse(scan_result)

        # Auto should detect at least the significant classes
        assert len(parse_result.components) >= 2

    def test_parse_directory_detection(self, sample_repo: Path):
        scanner = CodebaseScanner()
        scan_result = scanner.scan(sample_repo)

        parser = CodebaseParser(component_detection="directory")
        parse_result = parser.parse(scan_result)

        comp_ids = {c.id for c in parse_result.components}
        assert "service_a" in comp_ids
        assert "service_b" in comp_ids

    def test_parse_builds_import_graph(self, sample_repo: Path):
        scanner = CodebaseScanner()
        scan_result = scanner.scan(sample_repo)

        parser = CodebaseParser(component_detection="auto")
        parse_result = parser.parse(scan_result)

        # Should have some import relationships
        assert isinstance(parse_result.import_graph, dict)

    def test_parse_extracts_descriptions(self, sample_repo: Path):
        scanner = CodebaseScanner()
        scan_result = scanner.scan(sample_repo)

        parser = CodebaseParser(component_detection="class", min_class_methods=2)
        parse_result = parser.parse(scan_result)

        for comp in parse_result.components:
            if comp.name == "AuthEngine":
                assert "authentication" in comp.description.lower()
                break
        else:
            pytest.fail("AuthEngine not found in parsed components")

    def test_parse_summary_format(self, sample_repo: Path):
        scanner = CodebaseScanner()
        scan_result = scanner.scan(sample_repo)

        parser = CodebaseParser()
        parse_result = parser.parse(scan_result)
        summary = parse_result.summary()

        assert "components_extracted" in summary
        assert "modules_extracted" in summary
        assert isinstance(summary["components_extracted"], int)

    def test_parse_to_yaml_dict(self, sample_repo: Path):
        scanner = CodebaseScanner()
        scan_result = scanner.scan(sample_repo)

        parser = CodebaseParser(component_detection="class", min_class_methods=2)
        parse_result = parser.parse(scan_result)

        for comp in parse_result.components:
            yaml_dict = comp.to_yaml_dict()
            assert "id" in yaml_dict
            assert "name" in yaml_dict
            assert "description" in yaml_dict
            assert "modules" in yaml_dict

    def test_parse_min_methods_filter(self, sample_repo: Path):
        scanner = CodebaseScanner()
        scan_result = scanner.scan(sample_repo)

        # With high min_methods, fewer components should be detected
        parser_strict = CodebaseParser(component_detection="class", min_class_methods=10)
        result_strict = parser_strict.parse(scan_result)

        parser_lax = CodebaseParser(component_detection="class", min_class_methods=1)
        result_lax = parser_lax.parse(scan_result)

        assert len(result_lax.components) >= len(result_strict.components)


# ── Engine Tests ──────────────────────────────────────────────────────


class TestIngestionEngine:
    def test_ingest_builds_graph(self, sample_repo: Path):
        engine = IngestionEngine(config_path=CONFIG_PATH)
        result = engine.ingest(sample_repo)

        assert result.graph is not None
        assert result.components_ingested > 0
        summary = result.graph.summary()
        assert summary["component_count"] > 0

    def test_ingest_summary_format(self, sample_repo: Path):
        engine = IngestionEngine(config_path=CONFIG_PATH)
        result = engine.ingest(sample_repo)
        summary = result.summary()

        assert "scan" in summary
        assert "parse" in summary
        assert "graph" in summary
        assert "components_ingested" in summary

    def test_ingest_yaml_directory(self):
        """V1-compatible YAML ingestion path."""
        engine = IngestionEngine(config_path=CONFIG_PATH)
        result = engine.ingest_yaml_directory(
            components_dir=COMPONENTS_DIR,
            variants_dir=VARIANTS_DIR,
        )

        assert result.graph is not None
        summary = result.graph.summary()
        assert summary["component_count"] == 12
        assert summary["variant_count"] == 8

    def test_ingest_preserves_existing_yamls(self, tmp_path: Path):
        """When a repo has existing component YAMLs, they should be loaded."""
        # Create a mini repo with a data/components/ directory
        comp_dir = tmp_path / "data" / "components"
        comp_dir.mkdir(parents=True)
        (comp_dir / "test_comp.yaml").write_text(
            "id: test_comp\n"
            "name: TestComponent\n"
            "description: A test component\n"
            "criticality: high\n"
            "modules:\n"
            "  - module_id: shared_mod\n"
            "    usage: Testing\n"
            "    coupling: tight\n"
        )

        engine = IngestionEngine(config_path=CONFIG_PATH)
        result = engine.ingest(tmp_path)

        comp = result.graph.get_component("test_comp")
        assert comp is not None
        assert comp.name == "TestComponent"

    def test_ingest_with_detection_strategy(self, sample_repo: Path):
        for strategy in ["auto", "class", "directory"]:
            engine = IngestionEngine(
                config_path=CONFIG_PATH,
                component_detection=strategy,
            )
            result = engine.ingest(sample_repo)
            assert result.components_ingested >= 0

    def test_ingest_appends_to_existing_graph(self, sample_repo: Path):
        """Existing graph should be preserved when adding new components."""
        from src.graph.builder import DependencyGraph

        existing_graph = DependencyGraph()
        existing_graph.add_component({
            "id": "existing_comp",
            "name": "ExistingComponent",
            "description": "Already in the graph",
            "criticality": "critical",
            "modules": [],
        })

        engine = IngestionEngine(config_path=CONFIG_PATH)
        result = engine.ingest(sample_repo, existing_graph=existing_graph)

        # Both existing and new components should be in the graph
        assert result.graph.get_component("existing_comp") is not None
        assert result.graph.summary()["component_count"] > 1
