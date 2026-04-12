"""Dependency graph builder — constructs a bipartite graph from YAML definitions.

=== RAG Pipeline Learning: Why a Graph? ===

In a RAG system, you typically embed documents and retrieve by vector similarity.
But similarity search alone can't answer structural questions like "what breaks
if I change module X?" — that requires traversing explicit relationships.

This is why production RAG systems often combine:
  1. A VECTOR INDEX for semantic similarity ("find components similar to X")
  2. A KNOWLEDGE GRAPH for structural traversal ("find everything connected to X")

This module builds the knowledge graph half. Later, the impact analyzer will
merge graph traversal results with semantic retrieval results — a pattern called
"hybrid retrieval" or "GraphRAG."

=== Why Bipartite? ===

We model this as a BIPARTITE graph with two node types:
  - Component nodes (auth_engine, billing_core, ...)
  - Module nodes (crypto_utils, event_bus, ...)

Edges only connect components to modules (never component-to-component or
module-to-module directly). Two components are "related" when they share a
module — this is an IMPLICIT relationship derived from the graph structure.

This is more powerful than a simple adjacency list because:
  - It preserves WHY two components are related (the shared module)
  - It captures coupling strength per edge (tight/loose/optional)
  - It naturally supports multi-hop traversal (component → module → component → module → ...)
  - It makes cycle detection straightforward with standard graph algorithms
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import yaml


@dataclass
class ComponentData:
    """Parsed component definition from YAML."""

    id: str
    name: str
    description: str
    team_owner: str
    criticality: str
    modules: list[dict[str, str]]
    api_surface: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class VariantData:
    """Parsed variant definition from YAML."""

    id: str
    name: str
    tier: str
    region: str
    description: str
    max_seats: int
    components: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class DependencyGraph:
    """Builds and queries a bipartite dependency graph from YAML data.

    The graph has two node types:
      - 'component' nodes: represent platform components (AuthEngine, etc.)
      - 'module' nodes: represent shared modules (crypto_utils, etc.)

    Edges connect components to the modules they use, carrying metadata:
      - coupling: 'tight', 'loose', or 'optional'
      - usage: human-readable description of how the component uses the module

    === RAG Learning: Edge Metadata as Context ===

    The 'usage' field on each edge becomes valuable context when the LLM
    generates risk explanations. Instead of just saying "AuthEngine depends
    on crypto_utils," we can tell the LLM "AuthEngine uses crypto_utils for
    password hashing and JWT signing" — this is grounded generation, not
    hallucination. The graph stores the facts; the LLM explains them.
    """

    def __init__(
        self,
        components_dir: str | Path | None = None,
        variants_dir: str | Path | None = None,
    ) -> None:
        self._graph = nx.Graph()
        self._components: dict[str, ComponentData] = {}
        self._variants: dict[str, VariantData] = {}
        # Reverse index: component_id → list of variant_ids that include it
        self._component_to_variants: dict[str, list[str]] = {}

        if components_dir and variants_dir:
            self.load(components_dir, variants_dir)

    def load(
        self,
        components_dir: str | Path,
        variants_dir: str | Path,
    ) -> None:
        """Load YAML files and build the graph.

        This is intentionally separated from __init__ so you can also build
        the graph programmatically via add_component() for testing.
        """
        self._load_components(Path(components_dir))
        self._load_variants(Path(variants_dir))

    def _load_components(self, directory: Path) -> None:
        """Parse all component YAML files and add them to the graph."""
        for filepath in sorted(directory.glob("*.yaml")):
            with open(filepath) as f:
                raw = yaml.safe_load(f)
            self.add_component(raw)

    def _load_variants(self, directory: Path) -> None:
        """Parse all variant YAML files and build the variant index."""
        for filepath in sorted(directory.glob("*.yaml")):
            with open(filepath) as f:
                raw = yaml.safe_load(f)
            self.add_variant(raw)

    def add_component(self, raw: dict[str, Any]) -> None:
        """Add a single component to the graph from a parsed YAML dict.

        Creates a component node, module nodes (if new), and edges between them.
        Each edge carries coupling and usage metadata.
        """
        comp = ComponentData(
            id=raw["id"],
            name=raw["name"],
            description=raw.get("description", ""),
            team_owner=raw.get("team_owner", ""),
            criticality=raw.get("criticality", "medium"),
            modules=raw.get("modules", []),
            api_surface=raw.get("api_surface", []),
            raw=raw,
        )
        self._components[comp.id] = comp

        # Add component node with its metadata
        self._graph.add_node(
            comp.id,
            node_type="component",
            name=comp.name,
            description=comp.description,
            team_owner=comp.team_owner,
            criticality=comp.criticality,
        )

        # Add module nodes and edges
        for mod in comp.modules:
            module_id = mod["module_id"]
            coupling = mod.get("coupling", "tight")
            usage = mod.get("usage", "")

            # Module nodes are created on first encounter — no separate definition file needed
            if not self._graph.has_node(module_id):
                self._graph.add_node(module_id, node_type="module")

            # Edge carries the relationship metadata
            self._graph.add_edge(
                comp.id,
                module_id,
                coupling=coupling,
                usage=usage,
            )

    def add_variant(self, raw: dict[str, Any]) -> None:
        """Add a single variant to the variant index from a parsed YAML dict."""
        variant = VariantData(
            id=raw["id"],
            name=raw["name"],
            tier=raw.get("tier", ""),
            region=raw.get("region", ""),
            description=raw.get("description", ""),
            max_seats=raw.get("max_seats", -1),
            components=raw.get("components", []),
            metadata=raw.get("metadata", {}),
            raw=raw,
        )
        self._variants[variant.id] = variant

        # Build reverse index: component → variants
        for comp_entry in variant.components:
            comp_id = comp_entry["component_id"]
            if comp_id not in self._component_to_variants:
                self._component_to_variants[comp_id] = []
            self._component_to_variants[comp_id].append(variant.id)

    # ── Query Methods ─────────────────────────────────────────────────

    @property
    def graph(self) -> nx.Graph:
        """Access the underlying NetworkX graph for advanced queries."""
        return self._graph

    def get_components(self) -> dict[str, ComponentData]:
        """Return all component definitions, keyed by ID."""
        return dict(self._components)

    def get_variants(self) -> dict[str, VariantData]:
        """Return all variant definitions, keyed by ID."""
        return dict(self._variants)

    def get_component(self, component_id: str) -> ComponentData | None:
        """Look up a single component by ID."""
        return self._components.get(component_id)

    def get_variant(self, variant_id: str) -> VariantData | None:
        """Look up a single variant by ID."""
        return self._variants.get(variant_id)

    def get_component_modules(self, component_id: str) -> list[dict[str, str]]:
        """Get all modules used by a component, with coupling and usage metadata.

        Returns a list of dicts: [{'module_id': ..., 'coupling': ..., 'usage': ...}]
        """
        if component_id not in self._graph:
            return []

        result = []
        for neighbor in self._graph.neighbors(component_id):
            if self._graph.nodes[neighbor].get("node_type") == "module":
                edge_data = self._graph.edges[component_id, neighbor]
                result.append(
                    {
                        "module_id": neighbor,
                        "coupling": edge_data.get("coupling", "tight"),
                        "usage": edge_data.get("usage", ""),
                    }
                )
        return result

    def get_module_components(self, module_id: str) -> list[str]:
        """Get all components that use a given module.

        This is the key query for impact analysis: "if this module changes,
        which components are affected?"
        """
        if module_id not in self._graph:
            return []

        return [
            neighbor
            for neighbor in self._graph.neighbors(module_id)
            if self._graph.nodes[neighbor].get("node_type") == "component"
        ]

    def get_shared_modules(self) -> dict[str, list[str]]:
        """Find all modules shared by 2+ components.

        Returns {module_id: [component_id, ...]} for shared modules only.

        === RAG Learning: Why This Matters ===

        Shared modules are the "hidden connectors" in the dependency graph.
        They create implicit relationships between components that aren't
        obvious from looking at any single component's YAML file. This is
        exactly the kind of non-obvious relationship that makes graph
        traversal essential — vector similarity alone wouldn't catch these.
        """
        shared = {}
        for node, data in self._graph.nodes(data=True):
            if data.get("node_type") != "module":
                continue
            components = self.get_module_components(node)
            if len(components) >= 2:
                shared[node] = sorted(components)
        return shared

    def get_variants_for_component(self, component_id: str) -> list[str]:
        """Get all variant IDs that include a given component."""
        return list(self._component_to_variants.get(component_id, []))

    def get_edge_coupling(self, component_id: str, module_id: str) -> str | None:
        """Get the coupling strength between a component and module."""
        if self._graph.has_edge(component_id, module_id):
            return self._graph.edges[component_id, module_id].get("coupling")
        return None

    def get_edge_usage(self, component_id: str, module_id: str) -> str | None:
        """Get the usage description for a component-module relationship."""
        if self._graph.has_edge(component_id, module_id):
            return self._graph.edges[component_id, module_id].get("usage")
        return None

    # ── Graph Statistics ──────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return summary statistics about the graph.

        Useful for quick sanity checks and debugging.
        """
        component_nodes = [
            n
            for n, d in self._graph.nodes(data=True)
            if d.get("node_type") == "component"
        ]
        module_nodes = [
            n
            for n, d in self._graph.nodes(data=True)
            if d.get("node_type") == "module"
        ]
        return {
            "total_nodes": self._graph.number_of_nodes(),
            "component_count": len(component_nodes),
            "module_count": len(module_nodes),
            "edge_count": self._graph.number_of_edges(),
            "variant_count": len(self._variants),
            "shared_module_count": len(self.get_shared_modules()),
        }
