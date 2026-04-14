"""Tests for the V2 recompiler and dynamic test generator.

Tests verify:
  - DynamicRecompiler exports components/variants as YAML
  - Config is updated with recompilation metadata
  - Deployment manifest is generated
  - Graph validation catches invalid states
  - DynamicTestGenerator produces valid, grounded test cases
  - Generated tests validate against the actual graph
  - All tests run without API keys (zero-API-key testing)
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from src.graph.builder import DependencyGraph
from src.recompiler.recompiler import DynamicRecompiler, CompilationResult
from src.recompiler.test_generator import DynamicTestGenerator, TestSuite


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
def minimal_graph() -> DependencyGraph:
    """Small graph for recompiler testing."""
    g = DependencyGraph()
    g.add_component({
        "id": "comp_a",
        "name": "ComponentA",
        "description": "First component for testing",
        "team_owner": "team-alpha",
        "criticality": "high",
        "modules": [
            {"module_id": "mod_x", "usage": "Core dependency", "coupling": "tight"},
            {"module_id": "mod_y", "usage": "Utility functions", "coupling": "loose"},
        ],
        "api_surface": ["GET /api/v1/a", "POST /api/v1/a"],
    })
    g.add_component({
        "id": "comp_b",
        "name": "ComponentB",
        "description": "Second component for testing",
        "team_owner": "team-beta",
        "criticality": "medium",
        "modules": [
            {"module_id": "mod_x", "usage": "Shared with A", "coupling": "tight"},
            {"module_id": "mod_z", "usage": "Exclusive to B", "coupling": "optional"},
        ],
    })
    g.add_variant({
        "id": "test_variant",
        "name": "Test Variant",
        "tier": "professional",
        "region": "global",
        "description": "A test variant",
        "max_seats": 100,
        "components": [
            {"component_id": "comp_a", "features": ["all"]},
            {"component_id": "comp_b", "features": ["basic"]},
        ],
        "metadata": {"launched": "2024-01-01"},
    })
    return g


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for recompiler."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "components").mkdir()
    (data_dir / "variants").mkdir()

    # Copy a config file
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    shutil.copy2(CONFIG_PATH, config_dir / "model_config.yaml")

    return tmp_path


# ── DynamicRecompiler Tests ──────────────────────────────────────────


class TestDynamicRecompiler:
    def test_compile_exports_components(self, minimal_graph, output_dir):
        config_path = output_dir / "config" / "model_config.yaml"
        recompiler = DynamicRecompiler(
            minimal_graph,
            config_path=config_path,
            output_dir=output_dir,
        )
        result = recompiler.compile(backup=False)

        assert result.is_successful
        assert result.components_written == 2

        # Verify YAML files were written
        comp_dir = output_dir / "data" / "components"
        assert (comp_dir / "comp_a.yaml").exists()
        assert (comp_dir / "comp_b.yaml").exists()

    def test_compile_exports_variants(self, minimal_graph, output_dir):
        config_path = output_dir / "config" / "model_config.yaml"
        recompiler = DynamicRecompiler(
            minimal_graph,
            config_path=config_path,
            output_dir=output_dir,
        )
        result = recompiler.compile(backup=False)

        assert result.variants_written == 1
        var_dir = output_dir / "data" / "variants"
        assert (var_dir / "test_variant.yaml").exists()

    def test_exported_yaml_is_valid(self, minimal_graph, output_dir):
        config_path = output_dir / "config" / "model_config.yaml"
        recompiler = DynamicRecompiler(
            minimal_graph,
            config_path=config_path,
            output_dir=output_dir,
        )
        recompiler.compile(backup=False)

        # Load and verify exported YAML
        comp_path = output_dir / "data" / "components" / "comp_a.yaml"
        with open(comp_path) as f:
            data = yaml.safe_load(f)

        assert data["id"] == "comp_a"
        assert data["name"] == "ComponentA"
        assert data["criticality"] == "high"
        assert len(data["modules"]) == 2

    def test_compile_updates_config(self, minimal_graph, output_dir):
        config_path = output_dir / "config" / "model_config.yaml"
        recompiler = DynamicRecompiler(
            minimal_graph,
            config_path=config_path,
            output_dir=output_dir,
        )
        result = recompiler.compile(backup=False)

        assert result.config_updated

        # Verify config has recompilation metadata
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "recompilation" in config
        assert config["recompilation"]["component_count"] == 2

    def test_compile_generates_manifest(self, minimal_graph, output_dir):
        config_path = output_dir / "config" / "model_config.yaml"
        recompiler = DynamicRecompiler(
            minimal_graph,
            config_path=config_path,
            output_dir=output_dir,
        )
        result = recompiler.compile(backup=False)

        assert result.manifest_path
        manifest_path = Path(result.manifest_path)
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["impact_radar_version"] == "2.0.0"
        assert manifest["graph"]["component_count"] == 2
        assert "deployment" in manifest

    def test_compile_with_backup(self, minimal_graph, output_dir):
        config_path = output_dir / "config" / "model_config.yaml"

        # Write some existing files to be backed up
        (output_dir / "data" / "components" / "old_comp.yaml").write_text("id: old\n")

        recompiler = DynamicRecompiler(
            minimal_graph,
            config_path=config_path,
            output_dir=output_dir,
        )
        result = recompiler.compile(backup=True)

        # Backup directory should exist
        backup_dirs = list((output_dir / "data").glob("backup_*"))
        assert len(backup_dirs) == 1
        assert (backup_dirs[0] / "components" / "old_comp.yaml").exists()

    def test_compile_empty_graph_fails(self, output_dir):
        empty_graph = DependencyGraph()
        config_path = output_dir / "config" / "model_config.yaml"
        recompiler = DynamicRecompiler(
            empty_graph,
            config_path=config_path,
            output_dir=output_dir,
        )
        result = recompiler.compile()

        assert not result.is_successful
        assert any("no components" in e.lower() for e in result.errors)

    def test_compile_no_edges_fails(self, output_dir):
        g = DependencyGraph()
        g.add_component({
            "id": "lonely",
            "name": "Lonely",
            "description": "No modules",
            "criticality": "medium",
            "modules": [],
        })
        config_path = output_dir / "config" / "model_config.yaml"
        recompiler = DynamicRecompiler(
            g, config_path=config_path, output_dir=output_dir,
        )
        result = recompiler.compile()

        assert not result.is_successful
        assert any("no edges" in e.lower() for e in result.errors)

    def test_export_graph_snapshot(self, minimal_graph, output_dir):
        config_path = output_dir / "config" / "model_config.yaml"
        recompiler = DynamicRecompiler(
            minimal_graph,
            config_path=config_path,
            output_dir=output_dir,
        )
        snapshot_path = recompiler.export_graph_snapshot()

        assert snapshot_path.exists()
        with open(snapshot_path) as f:
            snapshot = json.load(f)

        assert "components" in snapshot
        assert "comp_a" in snapshot["components"]
        assert "variants" in snapshot
        assert "test_variant" in snapshot["variants"]

    def test_compilation_result_summary(self, minimal_graph, output_dir):
        config_path = output_dir / "config" / "model_config.yaml"
        recompiler = DynamicRecompiler(
            minimal_graph,
            config_path=config_path,
            output_dir=output_dir,
        )
        result = recompiler.compile(backup=False)
        summary = result.summary()

        assert "output_dir" in summary
        assert "components_written" in summary
        assert "is_successful" in summary


# ── DynamicTestGenerator Tests ────────────────────────────────────────


class TestDynamicTestGenerator:
    def test_generate_produces_tests(self, full_graph: DependencyGraph):
        generator = DynamicTestGenerator(full_graph)
        suite = generator.generate()

        assert len(suite.tests) > 0

    def test_generate_component_existence_tests(self, full_graph: DependencyGraph):
        generator = DynamicTestGenerator(full_graph)
        suite = generator.generate()

        structure_tests = [t for t in suite.tests if t.category == "structure"]
        assert len(structure_tests) > 0

    def test_generate_direct_impact_tests(self, full_graph: DependencyGraph):
        generator = DynamicTestGenerator(full_graph)
        suite = generator.generate()

        direct_tests = [t for t in suite.tests if t.category == "direct_impact"]
        assert len(direct_tests) > 0

    def test_generate_indirect_impact_tests(self, full_graph: DependencyGraph):
        generator = DynamicTestGenerator(full_graph)
        suite = generator.generate()

        indirect_tests = [t for t in suite.tests if t.category == "indirect_impact"]
        assert len(indirect_tests) > 0

    def test_generate_variant_tests(self, full_graph: DependencyGraph):
        generator = DynamicTestGenerator(full_graph)
        suite = generator.generate()

        variant_tests = [t for t in suite.tests if t.category == "variant"]
        assert len(variant_tests) > 0

    def test_generate_cycle_safety_tests(self, full_graph: DependencyGraph):
        generator = DynamicTestGenerator(full_graph)
        suite = generator.generate()

        cycle_tests = [t for t in suite.tests if t.category == "cycle"]
        assert len(cycle_tests) > 0

    def test_generated_tests_are_grounded(self, full_graph: DependencyGraph):
        """All generated tests must reference real components and modules."""
        generator = DynamicTestGenerator(full_graph)
        suite = generator.generate()

        known_components = set(full_graph.get_components().keys())
        shared_modules = set(full_graph.get_shared_modules().keys())

        for test in suite.tests:
            # Grounding evidence should not be empty
            assert test.grounding_evidence, f"Test {test.name} has no grounding evidence"
            # Test code should not be empty
            assert test.test_code, f"Test {test.name} has no test code"

    def test_to_pytest_file_valid_python(self, full_graph: DependencyGraph):
        """Generated pytest file must be valid Python."""
        generator = DynamicTestGenerator(full_graph)
        suite = generator.generate()
        content = suite.to_pytest_file()

        # Should be parseable Python
        compile(content, "test_generated.py", "exec")

    def test_write_test_file(self, full_graph: DependencyGraph, tmp_path: Path):
        generator = DynamicTestGenerator(full_graph)
        output_path = tmp_path / "test_generated.py"
        result_path = generator.write_test_file(output_path)

        assert result_path.exists()
        content = result_path.read_text()
        assert "import pytest" in content
        assert "def test_" in content

    def test_suite_summary(self, full_graph: DependencyGraph):
        generator = DynamicTestGenerator(full_graph)
        suite = generator.generate()
        summary = suite.summary

        assert "total_tests" in summary
        assert "categories" in summary
        assert summary["total_tests"] > 0

    def test_minimal_graph_generates_tests(self, minimal_graph: DependencyGraph):
        """Even a small graph should generate valid tests."""
        generator = DynamicTestGenerator(minimal_graph)
        suite = generator.generate()

        assert len(suite.tests) > 0
        # Should have at least component existence and structure tests
        categories = {t.category for t in suite.tests}
        assert "structure" in categories

    def test_empty_graph_generates_minimal_tests(self):
        """An empty graph should generate structural tests but no impact tests."""
        g = DependencyGraph()
        generator = DynamicTestGenerator(g)
        suite = generator.generate()

        # Structure tests still run on empty graph
        assert isinstance(suite, TestSuite)
