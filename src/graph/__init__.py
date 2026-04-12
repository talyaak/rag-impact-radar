"""Dependency graph construction and traversal.

Builds a bipartite graph (components <-> modules) from YAML definitions,
then traverses it to find all components affected by a change scope.
"""
