"""Codebase parser — extracts components, modules, and dependencies from source code.

Analyzes Python source files (via AST) and documentation to extract:
  - Class and module definitions (potential components)
  - Import relationships (potential module dependencies)
  - Docstrings and comments (descriptions for embeddings)
  - Function signatures (API surfaces)

The parser transforms raw source code into structured data that can populate
the bipartite dependency graph and vector store.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.ingestion.scanner import ScanResult, DiscoveredFile


@dataclass
class ExtractedComponent:
    """A component extracted from source code analysis."""

    id: str
    name: str
    description: str
    source_file: str
    team_owner: str = ""
    criticality: str = "medium"
    modules: list[dict[str, str]] = field(default_factory=list)
    api_surface: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    confidence: float = 1.0  # How confident we are this is a real component

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert to the YAML dict format expected by DependencyGraph.add_component()."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "team_owner": self.team_owner,
            "criticality": self.criticality,
            "modules": self.modules,
            "api_surface": self.api_surface,
        }


@dataclass
class ExtractedModule:
    """A shared module/library extracted from import analysis."""

    id: str
    name: str
    description: str
    source_file: str = ""
    used_by: list[str] = field(default_factory=list)  # component IDs


@dataclass
class ParseResult:
    """Complete parsing output from code analysis."""

    components: list[ExtractedComponent] = field(default_factory=list)
    modules: list[ExtractedModule] = field(default_factory=list)
    import_graph: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "components_extracted": len(self.components),
            "modules_extracted": len(self.modules),
            "import_edges": sum(len(v) for v in self.import_graph.values()),
            "warnings": len(self.warnings),
        }


def _to_snake_case(name: str) -> str:
    """Convert CamelCase or mixed to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _to_title_case(name: str) -> str:
    """Convert snake_case to Title Case."""
    return " ".join(word.capitalize() for word in name.split("_"))


class CodebaseParser:
    """Parses scanned source files to extract components and dependencies.

    Uses Python's AST module for reliable code analysis without execution.
    Falls back to regex-based heuristics for non-Python files.
    """

    def __init__(
        self,
        component_detection: str = "auto",
        min_class_methods: int = 2,
    ) -> None:
        """
        Args:
            component_detection: Strategy for identifying components.
                "auto" — use heuristics (classes with methods, service markers)
                "directory" — treat each top-level directory as a component
                "class" — treat each significant class as a component
            min_class_methods: Minimum methods for a class to be considered a component.
        """
        self._detection = component_detection
        self._min_methods = min_class_methods

    def parse(self, scan_result: ScanResult) -> ParseResult:
        """Parse all scanned files and extract structural information."""
        result = ParseResult()

        # Phase 1: Extract classes, functions, and imports from Python files
        file_analyses: dict[str, _FileAnalysis] = {}
        for pf in scan_result.python_files:
            analysis = self._analyze_python_file(pf)
            if analysis:
                file_analyses[pf.relative_path] = analysis

        # Phase 2: Identify components based on detection strategy
        if self._detection == "directory":
            result.components = self._detect_by_directory(
                scan_result, file_analyses
            )
        elif self._detection == "class":
            result.components = self._detect_by_class(file_analyses)
        else:  # "auto"
            result.components = self._detect_auto(
                scan_result, file_analyses
            )

        # Phase 3: Build import graph and identify shared modules
        result.import_graph = self._build_import_graph(file_analyses)
        result.modules = self._identify_shared_modules(
            result.components, result.import_graph, file_analyses
        )

        # Phase 4: Wire modules to components
        self._wire_modules_to_components(result.components, result.modules)

        # Phase 5: Extract API surfaces from FastAPI/Flask patterns
        self._extract_api_surfaces(result.components, file_analyses)

        return result

    def _analyze_python_file(self, discovered: DiscoveredFile) -> _FileAnalysis | None:
        """Parse a Python file using AST and extract structural info."""
        try:
            source = discovered.path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(discovered.path))
        except (SyntaxError, UnicodeDecodeError):
            return None

        analysis = _FileAnalysis(
            path=discovered.relative_path,
            filepath=discovered.path,
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                analysis.imports.append(module)
                for alias in node.names:
                    analysis.import_names.append(f"{module}.{alias.name}")

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                cls_info = _ClassInfo(
                    name=node.name,
                    docstring=ast.get_docstring(node) or "",
                    methods=[
                        n.name for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ],
                    decorators=[
                        self._decorator_name(d) for d in node.decorator_list
                    ],
                    line_number=node.lineno,
                )
                analysis.classes.append(cls_info)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = _FunctionInfo(
                    name=node.name,
                    docstring=ast.get_docstring(node) or "",
                    decorators=[
                        self._decorator_name(d) for d in node.decorator_list
                    ],
                    args=[a.arg for a in node.args.args if a.arg != "self"],
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    line_number=node.lineno,
                )
                analysis.functions.append(func_info)

        # Extract module-level docstring
        analysis.module_docstring = ast.get_docstring(tree) or ""

        return analysis

    def _decorator_name(self, node: ast.expr) -> str:
        """Extract decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._decorator_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._decorator_name(node.func)
        return ""

    def _detect_auto(
        self,
        scan_result: ScanResult,
        file_analyses: dict[str, _FileAnalysis],
    ) -> list[ExtractedComponent]:
        """Auto-detect components using heuristics.

        Priority:
        1. Service boundaries (directories with Dockerfile/package.json)
        2. Significant classes (with enough methods and a docstring)
        3. Top-level modules with route decorators (FastAPI/Flask)
        """
        components: list[ExtractedComponent] = []
        seen_ids: set[str] = set()

        # Strategy 1: Service boundaries
        for boundary in scan_result.service_boundaries:
            comp_id = _to_snake_case(boundary.name)
            if comp_id in seen_ids:
                continue
            seen_ids.add(comp_id)

            description = f"Service: {boundary.name}"
            # Try to find a main module docstring
            for py_file in boundary.python_files:
                rel = str(py_file.relative_to(scan_result.repo_root))
                if rel in file_analyses:
                    analysis = file_analyses[rel]
                    if analysis.module_docstring:
                        description = analysis.module_docstring.split("\n")[0]
                        break

            components.append(ExtractedComponent(
                id=comp_id,
                name=_to_title_case(comp_id),
                description=description,
                source_file=str(boundary.root_dir.relative_to(scan_result.repo_root)),
                confidence=0.9,
            ))

        # Strategy 2: Significant classes
        for rel_path, analysis in file_analyses.items():
            for cls in analysis.classes:
                if len(cls.methods) < self._min_methods:
                    continue
                if cls.name.startswith("_"):
                    continue

                comp_id = _to_snake_case(cls.name)
                if comp_id in seen_ids:
                    continue
                seen_ids.add(comp_id)

                description = cls.docstring.split("\n")[0] if cls.docstring else f"Class {cls.name}"

                components.append(ExtractedComponent(
                    id=comp_id,
                    name=cls.name,
                    description=description,
                    source_file=rel_path,
                    classes=[cls.name],
                    functions=cls.methods,
                    confidence=0.7,
                ))

        return components

    def _detect_by_directory(
        self,
        scan_result: ScanResult,
        file_analyses: dict[str, _FileAnalysis],
    ) -> list[ExtractedComponent]:
        """Treat each top-level directory as a component."""
        components: list[ExtractedComponent] = []
        top_dirs: set[str] = set()

        for pf in scan_result.python_files:
            parts = Path(pf.relative_path).parts
            if len(parts) > 1:
                top_dirs.add(parts[0])

        for dir_name in sorted(top_dirs):
            comp_id = _to_snake_case(dir_name)
            description = f"Directory module: {dir_name}"

            # Try to get description from __init__.py docstring
            init_path = f"{dir_name}/__init__.py"
            if init_path in file_analyses:
                doc = file_analyses[init_path].module_docstring
                if doc:
                    description = doc.split("\n")[0]

            components.append(ExtractedComponent(
                id=comp_id,
                name=_to_title_case(comp_id),
                description=description,
                source_file=dir_name,
                confidence=0.6,
            ))

        return components

    def _detect_by_class(
        self,
        file_analyses: dict[str, _FileAnalysis],
    ) -> list[ExtractedComponent]:
        """Treat each significant class as a component."""
        components: list[ExtractedComponent] = []
        seen_ids: set[str] = set()

        for rel_path, analysis in file_analyses.items():
            for cls in analysis.classes:
                if len(cls.methods) < self._min_methods:
                    continue
                comp_id = _to_snake_case(cls.name)
                if comp_id in seen_ids:
                    continue
                seen_ids.add(comp_id)

                description = cls.docstring.split("\n")[0] if cls.docstring else cls.name

                components.append(ExtractedComponent(
                    id=comp_id,
                    name=cls.name,
                    description=description,
                    source_file=rel_path,
                    classes=[cls.name],
                    functions=cls.methods,
                    confidence=0.8,
                ))

        return components

    def _build_import_graph(
        self,
        file_analyses: dict[str, _FileAnalysis],
    ) -> dict[str, list[str]]:
        """Build a file-to-file import dependency graph."""
        graph: dict[str, list[str]] = {}

        # Build a map of module names to files
        module_to_file: dict[str, str] = {}
        for rel_path in file_analyses:
            # Convert file path to Python module name
            mod_name = rel_path.replace("/", ".").replace("\\", ".")
            if mod_name.endswith(".py"):
                mod_name = mod_name[:-3]
            if mod_name.endswith(".__init__"):
                mod_name = mod_name[:-9]
            module_to_file[mod_name] = rel_path

        for rel_path, analysis in file_analyses.items():
            deps: list[str] = []
            for imp in analysis.imports:
                # Check if this import resolves to a file in the project
                for mod_name, mod_file in module_to_file.items():
                    if imp == mod_name or imp.startswith(mod_name + "."):
                        if mod_file != rel_path:
                            deps.append(mod_file)
                        break
            if deps:
                graph[rel_path] = sorted(set(deps))

        return graph

    def _identify_shared_modules(
        self,
        components: list[ExtractedComponent],
        import_graph: dict[str, list[str]],
        file_analyses: dict[str, _FileAnalysis],
    ) -> list[ExtractedModule]:
        """Identify files/modules imported by multiple components."""
        # Count how many components import each file
        file_importers: dict[str, set[str]] = {}

        comp_files = {c.source_file for c in components}

        for src_file, deps in import_graph.items():
            # Find which component this source file belongs to
            owner_comp = None
            for comp in components:
                if src_file == comp.source_file or src_file.startswith(comp.source_file + "/"):
                    owner_comp = comp.id
                    break

            if owner_comp is None:
                continue

            for dep_file in deps:
                if dep_file not in file_importers:
                    file_importers[dep_file] = set()
                file_importers[dep_file].add(owner_comp)

        # Files imported by 2+ components are shared modules
        shared_modules: list[ExtractedModule] = []
        for dep_file, importers in file_importers.items():
            if len(importers) < 2:
                continue

            mod_id = _to_snake_case(Path(dep_file).stem)
            description = ""
            if dep_file in file_analyses:
                doc = file_analyses[dep_file].module_docstring
                if doc:
                    description = doc.split("\n")[0]
            if not description:
                description = f"Shared module from {dep_file}"

            shared_modules.append(ExtractedModule(
                id=mod_id,
                name=_to_title_case(mod_id),
                description=description,
                source_file=dep_file,
                used_by=sorted(importers),
            ))

        return shared_modules

    def _wire_modules_to_components(
        self,
        components: list[ExtractedComponent],
        modules: list[ExtractedModule],
    ) -> None:
        """Connect extracted modules to their consuming components."""
        comp_map = {c.id: c for c in components}

        for mod in modules:
            for comp_id in mod.used_by:
                if comp_id in comp_map:
                    comp_map[comp_id].modules.append({
                        "module_id": mod.id,
                        "usage": mod.description,
                        "coupling": "tight",  # Default; gap analysis will refine
                    })

    def _extract_api_surfaces(
        self,
        components: list[ExtractedComponent],
        file_analyses: dict[str, _FileAnalysis],
    ) -> None:
        """Extract API endpoints from FastAPI/Flask route decorators."""
        for comp in components:
            if comp.source_file not in file_analyses:
                continue

            analysis = file_analyses[comp.source_file]

            for func in analysis.functions:
                for dec in func.decorators:
                    # FastAPI patterns: app.get, app.post, router.get, etc.
                    if any(method in dec.lower() for method in
                           [".get", ".post", ".put", ".delete", ".patch"]):
                        method = dec.split(".")[-1].upper()
                        endpoint = f"{method} /{func.name}"
                        comp.api_surface.append(endpoint)

            for cls in analysis.classes:
                for dec in cls.decorators:
                    if "controller" in dec.lower() or "resource" in dec.lower():
                        for method_name in cls.methods:
                            comp.api_surface.append(f"METHOD /{method_name}")


# ── Internal data classes for AST analysis ──────────────────────────────


@dataclass
class _ClassInfo:
    name: str
    docstring: str
    methods: list[str]
    decorators: list[str]
    line_number: int = 0


@dataclass
class _FunctionInfo:
    name: str
    docstring: str
    decorators: list[str]
    args: list[str]
    is_async: bool = False
    line_number: int = 0


@dataclass
class _FileAnalysis:
    path: str
    filepath: Path
    module_docstring: str = ""
    classes: list[_ClassInfo] = field(default_factory=list)
    functions: list[_FunctionInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    import_names: list[str] = field(default_factory=list)
