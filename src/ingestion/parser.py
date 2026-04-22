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
import fnmatch
import json
import re
import xml.etree.ElementTree as ET
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
    language: str = "python"  # Default preserves backward compat with existing callers

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


def _npm_id(raw: str) -> str:
    """Filesystem-safe component/module ID derived from an npm name.

    Scoped names like `@acme/auth-ui` become `acme__auth_ui`. Regular
    names like `react-router` become `react_router`. The double
    underscore is preserved as the scope separator so two distinct
    packages don't collide on their short names.
    """
    cleaned = raw.strip()
    if cleaned.startswith("@") and "/" in cleaned:
        scope, name = cleaned[1:].split("/", 1)
        scope = re.sub(r"[^A-Za-z0-9]+", "_", scope).strip("_").lower()
        name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
        return f"{scope}__{name}" if scope and name else (scope or name or "package")
    return re.sub(r"[^A-Za-z0-9]+", "_", cleaned).strip("_").lower() or "package"


def _is_internal_npm_dep(version_spec: str) -> bool:
    """True for workspace protocol or file/path references."""
    return (
        version_spec.startswith("workspace:")
        or version_spec.startswith("file:")
        or version_spec.startswith("link:")
        or version_spec.startswith("./")
        or version_spec.startswith("../")
    )


def _extract_package_json_author(data: dict[str, Any]) -> str:
    """Pull an author/team string from a package.json (best-effort)."""
    author = data.get("author")
    if isinstance(author, dict):
        name = author.get("name")
        if isinstance(name, str) and name:
            return name
    if isinstance(author, str) and author:
        return author.split("<")[0].strip()
    contributors = data.get("contributors")
    if isinstance(contributors, list) and contributors:
        first = contributors[0]
        if isinstance(first, dict):
            name = first.get("name")
            if isinstance(name, str) and name:
                return name
        if isinstance(first, str) and first:
            return first.split("<")[0].strip()
    return ""


def _infer_ts_or_js(package_json_path: Path, data: dict[str, Any]) -> str:
    """Return "typescript" if there are TS signals, else "javascript"."""
    if (package_json_path.parent / "tsconfig.json").exists():
        return "typescript"
    if data.get("types") or data.get("typings"):
        return "typescript"
    return "javascript"


def _csproj_id(raw: str) -> str:
    """Filesystem-safe ID for a .NET assembly / project name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower() or "project"


def _local_tag(elem: ET.Element) -> str:
    """Return the tag without its XML namespace prefix."""
    tag = elem.tag
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _iter_local(parent: ET.Element, name: str):
    """Yield direct children of ``parent`` whose local tag matches ``name``."""
    for child in parent:
        if _local_tag(child) == name:
            yield child


def _find_first_text(root: ET.Element, tag: str) -> str:
    """First non-empty <tag> text anywhere under ``root`` (namespace-agnostic)."""
    for elem in root.iter():
        if _local_tag(elem) == tag:
            text = (elem.text or "").strip()
            if text:
                return text
    return ""


def _csproj_relative_id(ref_path: str, csproj_path: Path, repo_root: Path) -> tuple[str, str]:
    """Given a ProjectReference Include="../Foo/Foo.csproj", return (id, rel).

    ``id`` is derived from the referenced project's filename stem.
    ``rel`` is a repo-relative string path to the referenced project,
    or the original (normalized) string if we can't resolve it.
    """
    normalized = ref_path.replace("\\", "/")
    stem = Path(normalized).stem or normalized
    try:
        resolved = (csproj_path.parent / normalized).resolve()
        rel = str(resolved.relative_to(repo_root))
    except (OSError, ValueError):
        rel = normalized
    return _csproj_id(stem), rel


_SLN_PROJECT_RE = re.compile(
    r'^\s*Project\("\{[0-9A-Fa-f-]+\}"\)\s*=\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"\{[0-9A-Fa-f-]+\}"',
    re.MULTILINE,
)


def _extract_workspace_patterns(package_json_data: dict[str, Any]) -> list[str]:
    """Return the workspace glob patterns declared by a package.json.

    Supports both the npm array form (``"workspaces": ["packages/*"]``)
    and the yarn object form (``"workspaces": {"packages": [...]}``).
    Returns an empty list when no valid patterns are declared.
    """
    ws = package_json_data.get("workspaces")
    if isinstance(ws, list):
        return [p for p in ws if isinstance(p, str)]
    if isinstance(ws, dict):
        packages = ws.get("packages")
        if isinstance(packages, list):
            return [p for p in packages if isinstance(p, str)]
    return []


def _merge_module_refs(target: "ExtractedComponent", source: "ExtractedComponent") -> None:
    """Merge ``source.modules`` into ``target.modules`` without duplicating ids."""
    existing = {m["module_id"] for m in target.modules}
    for ref in source.modules:
        if ref.get("module_id") in existing:
            continue
        target.modules.append(ref)
        existing.add(ref.get("module_id", ""))


class CodebaseParser:
    """Parses scanned source files to extract components and dependencies.

    Uses Python's AST module for reliable code analysis without execution.
    Falls back to regex-based heuristics for non-Python files.
    """

    def __init__(
        self,
        component_detection: str = "auto",
        min_class_methods: int = 2,
        languages: set[str] | None = None,
        artifact_granularity: str = "manifest",
    ) -> None:
        """
        Args:
            component_detection: Strategy for identifying components.
                "auto" — use heuristics (classes with methods, service markers)
                "directory" — treat each top-level directory as a component
                "class" — treat each significant class as a component
            min_class_methods: Minimum methods for a class to be considered a component.
            languages: Which language manifests to parse. Defaults to all supported
                ({"python", "typescript", "javascript", "csharp"}). An empty set
                disables all manifest parsers and falls back to Python-only.
            artifact_granularity: Coarseness of manifest-derived components.
                "manifest" (default) emits one component per package.json / .csproj.
                "app" collapses nested workspace packages into their root manifest.
                "service" aggregates all manifests inside a service boundary into one.
        """
        self._detection = component_detection
        self._min_methods = min_class_methods
        self._languages = (
            {"python", "typescript", "javascript", "csharp"}
            if languages is None
            else set(languages)
        )
        if artifact_granularity not in ("manifest", "app", "service"):
            raise ValueError(
                f"artifact_granularity must be one of 'manifest', 'app', 'service'; "
                f"got {artifact_granularity!r}"
            )
        self._granularity = artifact_granularity

    def parse(self, scan_result: ScanResult) -> ParseResult:
        """Parse all scanned files and extract structural information."""
        result = ParseResult()

        # Phase 1: Extract classes, functions, and imports from Python files
        file_analyses: dict[str, _FileAnalysis] = {}
        if "python" in self._languages:
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

        # Phase 6: Extract components from non-Python manifest files
        self._extract_from_manifests(scan_result, result)

        # Phase 7: Coarsen manifest-level components per artifact_granularity
        if self._granularity != "manifest":
            self._apply_granularity(scan_result, result)

        return result

    def _extract_from_manifests(
        self,
        scan_result: ScanResult,
        result: ParseResult,
    ) -> None:
        """Append manifest-derived components/modules to ``result``.

        Runs after Python extraction so manifest components can't
        shadow class-level ones. IDs that collide with an existing
        component are dropped with a warning; duplicate modules merge
        their ``used_by`` lists instead of being emitted twice.
        """
        repo_root = scan_result.repo_root
        existing_ids = {c.id for c in result.components}
        modules_by_id = {m.id: m for m in result.modules}

        def _add_component(
            component: ExtractedComponent,
            modules: list[ExtractedModule],
        ) -> None:
            if component.id in existing_ids:
                result.warnings.append(
                    f"Manifest component '{component.id}' "
                    f"({component.source_file}) collides with an existing "
                    f"component id — skipping."
                )
                return
            existing_ids.add(component.id)
            result.components.append(component)
            for mod in modules:
                prior = modules_by_id.get(mod.id)
                if prior is None:
                    modules_by_id[mod.id] = mod
                    result.modules.append(mod)
                else:
                    for user in mod.used_by:
                        if user not in prior.used_by:
                            prior.used_by.append(user)

        package_jsons = scan_result.manifests_by_kind.get("package_json", [])
        if package_jsons and self._languages & {"typescript", "javascript"}:
            for manifest in package_jsons:
                extracted = self._analyze_package_json(manifest, repo_root)
                if extracted is None:
                    result.warnings.append(
                        f"Could not parse package.json at {manifest.relative_path}"
                    )
                    continue
                component, modules = extracted
                if component.language not in self._languages:
                    continue
                _add_component(component, modules)

        csprojs = scan_result.manifests_by_kind.get("csproj", [])
        if csprojs and "csharp" in self._languages:
            for manifest in csprojs:
                extracted = self._analyze_csproj(manifest, repo_root)
                if extracted is None:
                    result.warnings.append(
                        f"Could not parse .csproj at {manifest.relative_path}"
                    )
                    continue
                component, modules = extracted
                _add_component(component, modules)

        # .sln grouping: parse to surface malformed solutions as warnings.
        # Component-level grouping (for artifact_granularity="service")
        # ships in Chunk 5.
        slns = scan_result.manifests_by_kind.get("sln", [])
        if slns and "csharp" in self._languages:
            for manifest in slns:
                if self._analyze_sln(manifest, repo_root) is None:
                    result.warnings.append(
                        f"Could not parse .sln at {manifest.relative_path}"
                    )

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

    def _analyze_package_json(
        self,
        discovered: DiscoveredFile,
        repo_root: Path,
    ) -> tuple[ExtractedComponent, list[ExtractedModule]] | None:
        """Extract a component from a package.json manifest.

        Returns (component, modules) tuple, or None if the manifest is
        unreadable. Deps become tight-coupled modules; devDeps become
        loose-coupled modules. `workspace:*` and relative-path deps are
        recognized as internal. Label is `"typescript"` when a sibling
        tsconfig.json exists or the package has a `"types"`/`"typings"`
        field; otherwise `"javascript"`.
        """
        try:
            data = json.loads(discovered.path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None

        raw_name = str(data.get("name") or discovered.path.parent.name or "package")
        comp_id = _npm_id(raw_name)
        description = str(data.get("description") or "").strip()
        confidence = 1.0
        if not description:
            description = f"{raw_name} npm package"
            confidence = 0.5

        component = ExtractedComponent(
            id=comp_id,
            name=raw_name,
            description=description.split("\n")[0][:500],
            source_file=discovered.relative_path,
            team_owner=_extract_package_json_author(data),
            api_surface=sorted({
                str(k) for k in (data.get("keywords") or []) if isinstance(k, str)
            }),
            confidence=confidence,
            language=_infer_ts_or_js(discovered.path, data),
        )

        modules: list[ExtractedModule] = []
        for dep_group, coupling in (("dependencies", "tight"), ("devDependencies", "loose")):
            deps = data.get(dep_group) or {}
            if not isinstance(deps, dict):
                continue
            for dep_name, dep_version in deps.items():
                if not isinstance(dep_name, str):
                    continue
                version = str(dep_version)
                is_internal = _is_internal_npm_dep(version)
                mod_id = _npm_id(dep_name)
                usage = (
                    f"Internal workspace dep ({version})"
                    if is_internal
                    else f"npm {dep_group} ({version})"
                )
                component.modules.append({
                    "module_id": mod_id,
                    "usage": usage,
                    "coupling": coupling,
                })
                modules.append(ExtractedModule(
                    id=mod_id,
                    name=dep_name,
                    description=(
                        f"Internal workspace package {dep_name}"
                        if is_internal
                        else f"External npm package {dep_name}"
                    ),
                    source_file=version if is_internal else "",
                    used_by=[comp_id],
                ))
        return component, modules

    def _apply_granularity(
        self,
        scan_result: ScanResult,
        result: ParseResult,
    ) -> None:
        """Collapse manifest-level components per ``artifact_granularity``.

        Both "app" and "service" are reductions on top of the manifest-level
        output. Python components are never touched — this only operates on
        manifest-derived components (language in {typescript, javascript, csharp}).
        """
        if self._granularity == "app":
            self._collapse_by_workspaces(scan_result, result)
        elif self._granularity == "service":
            self._collapse_by_service_boundary(scan_result, result)

    def _collapse_by_workspaces(
        self,
        scan_result: ScanResult,
        result: ParseResult,
    ) -> None:
        """Merge nested workspace package.json components into their root.

        A "root" is any package.json whose JSON declares a top-level
        ``workspaces`` field (npm/yarn workspace protocol). Nested
        package.json manifests whose directory matches one of the root's
        workspace glob patterns get their modules merged into the root
        component and are dropped from the component list.
        """
        repo_root = scan_result.repo_root
        roots: dict[Path, tuple[ExtractedComponent, list[str]]] = {}

        for comp in result.components:
            if comp.language not in {"javascript", "typescript"}:
                continue
            manifest_path = repo_root / comp.source_file
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            patterns = _extract_workspace_patterns(data)
            if patterns:
                roots[manifest_path.parent.resolve()] = (comp, patterns)

        if not roots:
            return

        kept: list[ExtractedComponent] = []
        for comp in result.components:
            if comp.language not in {"javascript", "typescript"}:
                kept.append(comp)
                continue
            manifest_path = (repo_root / comp.source_file).resolve()
            if manifest_path.parent in roots:
                # Component is itself a workspace root — keep it.
                kept.append(comp)
                continue
            matched: ExtractedComponent | None = None
            for root_dir, (root_comp, patterns) in roots.items():
                try:
                    rel = manifest_path.parent.relative_to(root_dir)
                except ValueError:
                    continue
                rel_str = str(rel).replace("\\", "/")
                for pattern in patterns:
                    pattern_norm = pattern.replace("\\", "/").rstrip("/")
                    if fnmatch.fnmatch(rel_str, pattern_norm) or fnmatch.fnmatch(
                        rel_str, pattern_norm + "/*"
                    ):
                        matched = root_comp
                        break
                if matched is not None:
                    break
            if matched is None:
                kept.append(comp)
            else:
                # Merge this component's modules into the matched root.
                _merge_module_refs(matched, comp)

        # Rewire ExtractedModule.used_by so dropped comp ids no longer appear.
        dropped_ids = {c.id for c in result.components} - {c.id for c in kept}
        if dropped_ids:
            for module in result.modules:
                module.used_by = [u for u in module.used_by if u not in dropped_ids]

        result.components = kept

    def _collapse_by_service_boundary(
        self,
        scan_result: ScanResult,
        result: ParseResult,
    ) -> None:
        """Merge all manifest components within a service boundary into one.

        For each ``ServiceBoundary`` discovered by the scanner, pick a lead
        component (the one whose source_file is closest to the boundary root)
        and merge the modules of any other manifest components inside the
        boundary into it. Python components are not touched.
        """
        repo_root = scan_result.repo_root
        manifest_comps_by_id = {
            c.id: c for c in result.components
            if c.language in {"javascript", "typescript", "csharp"}
        }
        if not manifest_comps_by_id:
            return

        absorbed_ids: set[str] = set()
        for boundary in scan_result.service_boundaries:
            members: list[ExtractedComponent] = []
            for comp in manifest_comps_by_id.values():
                if comp.id in absorbed_ids:
                    continue
                comp_dir = (repo_root / comp.source_file).parent.resolve()
                try:
                    comp_dir.relative_to(boundary.root_dir.resolve())
                except ValueError:
                    continue
                members.append(comp)
            if len(members) < 2:
                continue
            # Lead: the manifest closest to the boundary root.
            members.sort(
                key=lambda c: len((repo_root / c.source_file).parent.resolve().parts)
            )
            lead = members[0]
            for follower in members[1:]:
                _merge_module_refs(lead, follower)
                absorbed_ids.add(follower.id)

        if absorbed_ids:
            result.components = [c for c in result.components if c.id not in absorbed_ids]
            for module in result.modules:
                module.used_by = [u for u in module.used_by if u not in absorbed_ids]

    def _analyze_csproj(
        self,
        discovered: DiscoveredFile,
        repo_root: Path,
    ) -> tuple[ExtractedComponent, list[ExtractedModule]] | None:
        """Extract a component from a .NET .csproj manifest.

        Handles both SDK-style (no xmlns) and legacy (xmlns) projects by
        matching local tag names only. ``<ProjectReference>`` entries
        become internal modules; ``<PackageReference>`` entries become
        external modules. Component language is labeled ``"csharp"``.
        Returns None for unreadable or malformed XML.
        """
        try:
            tree = ET.parse(discovered.path)
        except (ET.ParseError, OSError):
            return None
        root = tree.getroot()

        assembly = _find_first_text(root, "AssemblyName")
        package_id = _find_first_text(root, "PackageId")
        project_stem = discovered.path.stem
        raw_name = assembly or package_id or project_stem
        comp_id = _csproj_id(raw_name)

        description = _find_first_text(root, "Description")
        confidence = 1.0
        if not description:
            description = f"{raw_name} .NET project"
            confidence = 0.5

        authors = _find_first_text(root, "Authors")
        team_owner = authors.split(",")[0].strip() if authors else ""

        tags_raw = _find_first_text(root, "PackageTags")
        api_surface = sorted({t for t in re.split(r"[;,\s]+", tags_raw) if t}) if tags_raw else []

        component = ExtractedComponent(
            id=comp_id,
            name=raw_name,
            description=description.split("\n")[0][:500],
            source_file=discovered.relative_path,
            team_owner=team_owner,
            api_surface=api_surface,
            confidence=confidence,
            language="csharp",
        )

        modules: list[ExtractedModule] = []

        for ref in root.iter():
            tag = _local_tag(ref)
            if tag == "ProjectReference":
                include = ref.get("Include") or ""
                if not include:
                    continue
                mod_id, rel = _csproj_relative_id(include, discovered.path, repo_root)
                component.modules.append({
                    "module_id": mod_id,
                    "usage": f"ProjectReference ({rel})",
                    "coupling": "tight",
                })
                modules.append(ExtractedModule(
                    id=mod_id,
                    name=Path(include.replace("\\", "/")).stem or include,
                    description=f"Internal .NET project referenced at {rel}",
                    source_file=rel,
                    used_by=[comp_id],
                ))
            elif tag == "PackageReference":
                include = ref.get("Include") or ""
                if not include:
                    continue
                version = ref.get("Version") or _find_first_text(ref, "Version") or ""
                mod_id = _csproj_id(include)
                component.modules.append({
                    "module_id": mod_id,
                    "usage": f"PackageReference ({version})" if version else "PackageReference",
                    "coupling": "tight",
                })
                modules.append(ExtractedModule(
                    id=mod_id,
                    name=include,
                    description=f"External NuGet package {include}",
                    source_file="",
                    used_by=[comp_id],
                ))

        return component, modules

    def _analyze_sln(
        self,
        discovered: DiscoveredFile,
        repo_root: Path,
    ) -> tuple[str, list[str]] | None:
        """Parse a .sln file to extract the list of .csproj paths it groups.

        Returns ``(solution_name, csproj_relative_paths)`` or None if the
        file is unreadable. Only ``.csproj`` entries are returned — the
        .sln format also lists solution folders and other project types,
        which we deliberately skip.
        """
        try:
            text = discovered.path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            return None

        csproj_paths: list[str] = []
        for match in _SLN_PROJECT_RE.finditer(text):
            _, project_path = match.group(1), match.group(2)
            normalized = project_path.replace("\\", "/")
            if not normalized.lower().endswith(".csproj"):
                continue
            try:
                resolved = (discovered.path.parent / normalized).resolve()
                rel = str(resolved.relative_to(repo_root))
            except (OSError, ValueError):
                rel = normalized
            csproj_paths.append(rel)

        return discovered.path.stem, csproj_paths

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
