"""Dynamic recompiler — persists curated graph and recompiles for deployment.

After ingestion and gap analysis refine the dependency graph, the recompiler:
  1. Exports the graph as component/variant YAML files
  2. Persists embeddings in the vector store
  3. Updates model_config.yaml with tuned parameters
  4. Generates a deployment manifest
  5. Validates the compiled state is production-ready

The recompiler transforms the in-memory curated state into a durable,
deployment-ready artifact that can be served by the FastAPI layer.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.graph.builder import DependencyGraph
from src.rag.vector_store import VectorStore
from src.rag.embedder import ComponentEmbedder
from src.gap_analysis.detector import GapReport

logger = logging.getLogger(__name__)


@dataclass
class CompilationResult:
    """Output of the recompilation process."""

    output_dir: Path
    components_written: int = 0
    variants_written: int = 0
    documents_embedded: int = 0
    config_updated: bool = False
    manifest_path: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        return len(self.errors) == 0 and self.components_written > 0

    def summary(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "components_written": self.components_written,
            "variants_written": self.variants_written,
            "documents_embedded": self.documents_embedded,
            "config_updated": self.config_updated,
            "manifest_path": self.manifest_path,
            "is_successful": self.is_successful,
            "errors": self.errors,
        }


class DynamicRecompiler:
    """Persists the curated graph and recompiles the system for deployment.

    The recompiler ensures that the in-memory state produced by ingestion
    and gap analysis is durably saved and the system is ready to serve
    impact analysis requests against the user's actual codebase.
    """

    def __init__(
        self,
        graph: DependencyGraph,
        config_path: str | Path = "config/model_config.yaml",
        output_dir: str | Path | None = None,
    ) -> None:
        self._graph = graph
        self._config_path = Path(config_path)
        self._output_dir = Path(output_dir) if output_dir else self._config_path.parent.parent

    def compile(
        self,
        vector_store: VectorStore | None = None,
        embedding_fn: Any = None,
        gap_report: GapReport | None = None,
        backup: bool = True,
    ) -> CompilationResult:
        """Run the full recompilation pipeline.

        Args:
            vector_store: Vector store to rebuild embeddings in.
            embedding_fn: Optional embedding function.
            gap_report: Gap report to validate completeness.
            backup: If True, backup existing data before overwriting.

        Returns:
            CompilationResult with details of what was written.
        """
        result = CompilationResult(output_dir=self._output_dir)

        # Phase 1: Validate graph state
        validation_errors = self._validate_graph()
        if validation_errors:
            result.errors.extend(validation_errors)
            return result

        # Phase 2: Backup existing data
        if backup:
            self._backup_data()

        # Phase 3: Export components as YAML
        try:
            result.components_written = self._export_components()
        except Exception as e:
            result.errors.append(f"Component export failed: {e}")
            return result

        # Phase 4: Export variants as YAML
        try:
            result.variants_written = self._export_variants()
        except Exception as e:
            result.errors.append(f"Variant export failed: {e}")

        # Phase 5: Rebuild embeddings
        if vector_store is not None:
            try:
                result.documents_embedded = self._rebuild_embeddings(
                    vector_store, embedding_fn
                )
            except Exception as e:
                result.errors.append(f"Embedding rebuild failed: {e}")

        # Phase 6: Update config
        try:
            self._update_config(gap_report)
            result.config_updated = True
        except Exception as e:
            result.errors.append(f"Config update failed: {e}")

        # Phase 7: Generate manifest
        try:
            result.manifest_path = str(self._generate_manifest(result, gap_report))
        except Exception as e:
            result.errors.append(f"Manifest generation failed: {e}")

        logger.info("Recompilation complete: %s", result.summary())
        return result

    def _validate_graph(self) -> list[str]:
        """Validate the graph is ready for export."""
        errors: list[str] = []

        components = self._graph.get_components()
        if not components:
            errors.append("Graph has no components")
            return errors

        summary = self._graph.summary()
        if summary["edge_count"] == 0:
            errors.append("Graph has no edges (no component-module relationships)")

        # Check for unnamed components
        for comp_id, comp in components.items():
            if not comp.name or comp.name == comp_id:
                errors.append(f"Component '{comp_id}' has no proper name")

        return errors

    def _backup_data(self) -> None:
        """Backup existing component and variant YAML files."""
        components_dir = self._output_dir / "data" / "components"
        variants_dir = self._output_dir / "data" / "variants"

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = self._output_dir / "data" / f"backup_{timestamp}"

        for src_dir, name in [(components_dir, "components"), (variants_dir, "variants")]:
            if src_dir.exists() and list(src_dir.glob("*.yaml")):
                dest = backup_dir / name
                dest.mkdir(parents=True, exist_ok=True)
                for yaml_file in src_dir.glob("*.yaml"):
                    shutil.copy2(yaml_file, dest / yaml_file.name)
                logger.info("Backed up %s to %s", name, dest)

    def _export_components(self) -> int:
        """Export all components as individual YAML files."""
        components_dir = self._output_dir / "data" / "components"
        components_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for comp_id, comp in self._graph.get_components().items():
            yaml_data = {
                "id": comp.id,
                "name": comp.name,
                "description": comp.description,
                "team_owner": comp.team_owner,
                "criticality": comp.criticality,
                "modules": comp.modules,
                "api_surface": comp.api_surface,
            }

            filepath = components_dir / f"{comp_id}.yaml"
            with open(filepath, "w") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            count += 1

        logger.info("Exported %d component YAML files", count)
        return count

    def _export_variants(self) -> int:
        """Export all variants as individual YAML files."""
        variants_dir = self._output_dir / "data" / "variants"
        variants_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for variant_id, variant in self._graph.get_variants().items():
            yaml_data = {
                "id": variant.id,
                "name": variant.name,
                "tier": variant.tier,
                "region": variant.region,
                "description": variant.description,
                "max_seats": variant.max_seats,
                "components": variant.components,
                "metadata": variant.metadata,
            }

            filepath = variants_dir / f"{variant_id}.yaml"
            with open(filepath, "w") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            count += 1

        logger.info("Exported %d variant YAML files", count)
        return count

    def _rebuild_embeddings(
        self,
        vector_store: VectorStore,
        embedding_fn: Any = None,
    ) -> int:
        """Rebuild the vector store with current graph data."""
        embedder = ComponentEmbedder(
            self._graph,
            vector_store,
            config_path=self._config_path,
        )
        return embedder.embed_and_store(embedding_fn=embedding_fn, reset=True)

    def _update_config(self, gap_report: GapReport | None = None) -> None:
        """Update model_config.yaml with recompilation metadata."""
        config: dict[str, Any] = {}
        if self._config_path.exists():
            with open(self._config_path) as f:
                config = yaml.safe_load(f) or {}

        # Add recompilation metadata
        graph_summary = self._graph.summary()
        config["recompilation"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component_count": graph_summary["component_count"],
            "module_count": graph_summary["module_count"],
            "edge_count": graph_summary["edge_count"],
            "variant_count": graph_summary["variant_count"],
            "shared_module_count": graph_summary["shared_module_count"],
        }

        if gap_report is not None:
            config["recompilation"]["gap_analysis"] = {
                "completeness_score": gap_report.completeness_score,
                "total_gaps": len(gap_report.gaps),
                "resolved_gaps": sum(1 for g in gap_report.gaps if g.resolved),
            }

        # Adjust graph traversal depth based on graph size
        if graph_summary["component_count"] > 50:
            config.setdefault("graph", {})["max_traversal_depth"] = 3
        elif graph_summary["component_count"] > 20:
            config.setdefault("graph", {})["max_traversal_depth"] = 4

        with open(self._config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info("Updated config at %s", self._config_path)

    def _generate_manifest(
        self,
        result: CompilationResult,
        gap_report: GapReport | None = None,
    ) -> Path:
        """Generate a deployment manifest summarizing the compiled state."""
        manifest = {
            "impact_radar_version": "2.0.0",
            "compiled_at": datetime.now(timezone.utc).isoformat(),
            "graph": self._graph.summary(),
            "compilation": {
                "components_written": result.components_written,
                "variants_written": result.variants_written,
                "documents_embedded": result.documents_embedded,
                "config_updated": result.config_updated,
            },
            "deployment": {
                "api_endpoint": "/api/v1/analyze",
                "health_check": "/health",
                "data_directory": str(self._output_dir / "data"),
                "config_path": str(self._config_path),
            },
        }

        if gap_report:
            manifest["quality"] = {
                "completeness_score": gap_report.completeness_score,
                "total_gaps": len(gap_report.gaps),
                "unresolved_gaps": len(gap_report.unresolved),
                "critical_gaps": len(gap_report.critical_gaps),
            }

        manifest_path = self._output_dir / "deployment_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Generated deployment manifest at %s", manifest_path)
        return manifest_path

    def export_graph_snapshot(self, output_path: str | Path | None = None) -> Path:
        """Export the full graph state as a single JSON file for backup/transfer."""
        path = Path(output_path) if output_path else self._output_dir / "graph_snapshot.json"

        snapshot = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "summary": self._graph.summary(),
            "components": {},
            "variants": {},
        }

        for comp_id, comp in self._graph.get_components().items():
            snapshot["components"][comp_id] = {
                "id": comp.id,
                "name": comp.name,
                "description": comp.description,
                "team_owner": comp.team_owner,
                "criticality": comp.criticality,
                "modules": comp.modules,
                "api_surface": comp.api_surface,
            }

        for var_id, var in self._graph.get_variants().items():
            snapshot["variants"][var_id] = {
                "id": var.id,
                "name": var.name,
                "tier": var.tier,
                "region": var.region,
                "description": var.description,
                "components": var.components,
                "metadata": var.metadata,
            }

        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)

        return path
