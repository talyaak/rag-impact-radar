"""Ingestion engine — orchestrates scanning, parsing, graph building, and embedding.

This is the main entry point for V2 codebase ingestion. It coordinates:
  1. CodebaseScanner — discover files in the repository
  2. CodebaseParser — extract components, modules, and dependencies
  3. DependencyGraph — build the bipartite graph from extracted data
  4. ComponentEmbedder — embed descriptions into the vector store

The engine produces an IngestionResult that captures the full state of the
ingested codebase, ready for gap analysis and refinement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.graph.builder import DependencyGraph
from src.rag.vector_store import VectorStore
from src.rag.embedder import ComponentEmbedder
from src.ingestion.scanner import CodebaseScanner, ScanResult
from src.ingestion.parser import CodebaseParser, ParseResult, ExtractedComponent

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Complete output of the ingestion pipeline."""

    scan_result: ScanResult
    parse_result: ParseResult
    graph: DependencyGraph
    documents_embedded: int = 0
    components_ingested: int = 0
    modules_ingested: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "scan": self.scan_result.summary(),
            "parse": self.parse_result.summary(),
            "graph": self.graph.summary(),
            "documents_embedded": self.documents_embedded,
            "components_ingested": self.components_ingested,
            "modules_ingested": self.modules_ingested,
            "warnings": self.warnings,
        }


class IngestionEngine:
    """Orchestrates the full V2 ingestion pipeline.

    Takes a repository path and produces a populated DependencyGraph and
    vector store, ready for gap analysis and impact analysis.
    """

    def __init__(
        self,
        config_path: str | Path = "config/model_config.yaml",
        component_detection: str = "auto",
        persist_directory: str | None = None,
    ) -> None:
        self._config_path = Path(config_path)
        self._config = self._load_config()
        self._detection = component_detection

        ingestion_config = self._config.get("ingestion", {})
        self._min_class_methods = ingestion_config.get("min_class_methods", 2)
        self._max_file_size = ingestion_config.get("max_file_size_bytes", 5_000_000)
        self._persist_dir = persist_directory

    def _load_config(self) -> dict[str, Any]:
        if self._config_path.exists():
            with open(self._config_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def ingest(
        self,
        repo_path: str | Path,
        embed: bool = False,
        embedding_fn: Any = None,
        existing_graph: DependencyGraph | None = None,
        existing_store: VectorStore | None = None,
    ) -> IngestionResult:
        """Run the full ingestion pipeline on a repository.

        Args:
            repo_path: Path to the repository root.
            embed: If True, embed component descriptions into vector store.
            embedding_fn: Optional embedding function for vector store.
            existing_graph: Reuse an existing graph (append components).
            existing_store: Reuse an existing vector store.

        Returns:
            IngestionResult with all extracted data and built graph.
        """
        repo_root = Path(repo_path).resolve()
        logger.info("Starting ingestion of %s", repo_root)

        # Phase 1: Scan
        logger.info("Phase 1: Scanning repository")
        scanner = CodebaseScanner(max_file_size_bytes=self._max_file_size)
        scan_result = scanner.scan(repo_root)
        logger.info("Scan complete: %s", scan_result.summary())

        # Phase 2: Parse
        logger.info("Phase 2: Parsing source files")
        parser = CodebaseParser(
            component_detection=self._detection,
            min_class_methods=self._min_class_methods,
        )
        parse_result = parser.parse(scan_result)
        logger.info("Parse complete: %s", parse_result.summary())

        # Phase 3: Build graph
        logger.info("Phase 3: Building dependency graph")
        graph = existing_graph or DependencyGraph()

        # If there are existing component YAMLs, load them first
        if scan_result.component_yamls:
            logger.info(
                "Found %d existing component YAMLs, loading...",
                len(scan_result.component_yamls),
            )
            for yaml_path in scan_result.component_yamls:
                with open(yaml_path) as f:
                    raw = yaml.safe_load(f)
                if raw and "id" in raw:
                    graph.add_component(raw)

        if scan_result.variant_yamls:
            for yaml_path in scan_result.variant_yamls:
                with open(yaml_path) as f:
                    raw = yaml.safe_load(f)
                if raw and "id" in raw:
                    graph.add_variant(raw)

        # Add parsed components (skip duplicates)
        existing_ids = set(graph.get_components().keys())
        new_count = 0
        for comp in parse_result.components:
            if comp.id not in existing_ids:
                graph.add_component(comp.to_yaml_dict())
                new_count += 1
                existing_ids.add(comp.id)

        logger.info("Graph built: %d new components added", new_count)

        # Phase 4: Embed (optional)
        docs_embedded = 0
        if embed:
            logger.info("Phase 4: Embedding component descriptions")
            store = existing_store or VectorStore(
                config_path=self._config_path,
                persist_directory=self._persist_dir,
            )
            embedder = ComponentEmbedder(
                graph, store, config_path=self._config_path
            )
            docs_embedded = embedder.embed_and_store(
                embedding_fn=embedding_fn, reset=True
            )
            logger.info("Embedded %d documents", docs_embedded)

        # Collect warnings
        warnings = list(parse_result.warnings)
        orphan_modules = self._find_orphan_modules(parse_result)
        if orphan_modules:
            warnings.append(
                f"Found {len(orphan_modules)} modules used by only 1 component: "
                f"{', '.join(orphan_modules[:5])}"
            )

        low_confidence = [
            c for c in parse_result.components if c.confidence < 0.5
        ]
        if low_confidence:
            warnings.append(
                f"{len(low_confidence)} low-confidence components detected"
            )

        return IngestionResult(
            scan_result=scan_result,
            parse_result=parse_result,
            graph=graph,
            documents_embedded=docs_embedded,
            components_ingested=len(parse_result.components),
            modules_ingested=len(parse_result.modules),
            warnings=warnings,
        )

    def ingest_yaml_directory(
        self,
        components_dir: str | Path,
        variants_dir: str | Path | None = None,
        embed: bool = False,
        embedding_fn: Any = None,
        vector_store: VectorStore | None = None,
    ) -> IngestionResult:
        """Ingest from existing YAML definitions (V1-compatible path).

        This provides backward compatibility with the V1 workflow where
        components and variants are defined as YAML files.
        """
        graph = DependencyGraph()
        comp_dir = Path(components_dir)

        for yaml_path in sorted(comp_dir.glob("*.yaml")):
            with open(yaml_path) as f:
                raw = yaml.safe_load(f)
            if raw and "id" in raw:
                graph.add_component(raw)

        if variants_dir:
            var_dir = Path(variants_dir)
            for yaml_path in sorted(var_dir.glob("*.yaml")):
                with open(yaml_path) as f:
                    raw = yaml.safe_load(f)
                if raw and "id" in raw:
                    graph.add_variant(raw)

        docs_embedded = 0
        if embed:
            store = vector_store or VectorStore(
                config_path=self._config_path,
                persist_directory=self._persist_dir,
            )
            embedder = ComponentEmbedder(
                graph, store, config_path=self._config_path
            )
            docs_embedded = embedder.embed_and_store(
                embedding_fn=embedding_fn, reset=True
            )

        # Create minimal scan/parse results for compatibility
        scan_result = ScanResult(repo_root=comp_dir.parent)
        scan_result.component_yamls = sorted(comp_dir.glob("*.yaml"))
        if variants_dir:
            scan_result.variant_yamls = sorted(Path(variants_dir).glob("*.yaml"))

        parse_result = ParseResult()

        return IngestionResult(
            scan_result=scan_result,
            parse_result=parse_result,
            graph=graph,
            documents_embedded=docs_embedded,
            components_ingested=len(graph.get_components()),
            modules_ingested=graph.summary()["module_count"],
        )

    def _find_orphan_modules(self, parse_result: ParseResult) -> list[str]:
        """Find modules used by only one component (potential orphans)."""
        return [
            m.id for m in parse_result.modules
            if len(m.used_by) < 2
        ]
