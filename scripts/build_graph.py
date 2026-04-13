#!/usr/bin/env python3
"""One-time script to build and inspect the dependency graph from YAML.

=== RAG Pipeline Learning: Graph as Knowledge Base ===

This script builds the dependency graph and prints a diagnostic summary.
It's the "structured knowledge base" half of the hybrid RAG system.

Usage:
    python scripts/build_graph.py
    python scripts/build_graph.py --show-shared-modules
    python scripts/build_graph.py --show-variants
    python scripts/build_graph.py --check-connectivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.builder import DependencyGraph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and inspect the dependency graph from YAML data."
    )
    parser.add_argument("--show-shared-modules", action="store_true",
                        help="Display all shared modules and their components.")
    parser.add_argument("--show-variants", action="store_true",
                        help="Display all variants and their components.")
    parser.add_argument("--check-connectivity", action="store_true",
                        help="Verify all components are reachable from any starting point.")
    parser.add_argument("--config", type=str, default="config/model_config.yaml")
    args = parser.parse_args()

    components_dir = PROJECT_ROOT / "data" / "components"
    variants_dir = PROJECT_ROOT / "data" / "variants"

    print("=" * 60)
    print("Impact Radar — Dependency Graph Builder")
    print("=" * 60)

    graph = DependencyGraph(components_dir, variants_dir)
    summary = graph.summary()

    print(f"\nGraph Summary:")
    print(f"  Components:     {summary['component_count']}")
    print(f"  Modules:        {summary['module_count']}")
    print(f"  Total nodes:    {summary['total_nodes']}")
    print(f"  Edges:          {summary['edge_count']}")
    print(f"  Variants:       {summary['variant_count']}")
    print(f"  Shared modules: {summary['shared_module_count']}")

    if args.show_shared_modules:
        print(f"\nShared Modules:")
        for mod_id, comps in sorted(graph.get_shared_modules().items()):
            print(f"  {mod_id}: {', '.join(sorted(comps))}")

    if args.show_variants:
        print(f"\nVariants:")
        for variant_id, variant in sorted(graph.get_variants().items()):
            comp_ids = [c["component_id"] for c in variant.components]
            print(f"  {variant.name} ({variant_id}): {len(comp_ids)} components")
            for cid in comp_ids:
                print(f"    - {cid}")

    if args.check_connectivity:
        import networkx as nx
        print(f"\nConnectivity Check:")
        if nx.is_connected(graph.graph):
            print("  Graph is fully connected — all components reachable from any starting point.")
        else:
            components = list(nx.connected_components(graph.graph))
            print(f"  WARNING: Graph has {len(components)} disconnected components:")
            for i, comp_set in enumerate(components):
                print(f"    Component {i}: {comp_set}")

    print()


if __name__ == "__main__":
    main()
