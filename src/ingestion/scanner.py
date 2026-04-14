"""Codebase scanner — discovers source files, documentation, and config across a repository.

Walks a user-provided repository root and identifies:
  - Python modules and packages
  - Service/microservice boundaries (Dockerfile, docker-compose, package.json, etc.)
  - Configuration files (YAML, TOML, JSON, .env)
  - Documentation (Markdown, RST, text)
  - Existing Impact Radar component/variant YAML files

The scanner produces a ScanResult that downstream parsers consume to extract
components, modules, and dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# File categories for classification
_PYTHON_EXTS = {".py"}
_CONFIG_EXTS = {".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".env"}
_DOC_EXTS = {".md", ".rst", ".txt"}
_SERVICE_MARKERS = {
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "package.json",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
    "setup.py",
    "pyproject.toml",
}

_DEFAULT_IGNORE = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
    "chroma_db",
}


@dataclass
class DiscoveredFile:
    """A single file discovered during scanning."""

    path: Path
    relative_path: str
    category: str  # "python", "config", "doc", "service_marker", "other"
    size_bytes: int = 0


@dataclass
class ServiceBoundary:
    """A detected service/microservice boundary within the repo."""

    name: str
    root_dir: Path
    marker_file: str
    python_files: list[Path] = field(default_factory=list)
    config_files: list[Path] = field(default_factory=list)
    doc_files: list[Path] = field(default_factory=list)


@dataclass
class ScanResult:
    """Complete scan output for a repository."""

    repo_root: Path
    python_files: list[DiscoveredFile] = field(default_factory=list)
    config_files: list[DiscoveredFile] = field(default_factory=list)
    doc_files: list[DiscoveredFile] = field(default_factory=list)
    service_boundaries: list[ServiceBoundary] = field(default_factory=list)
    component_yamls: list[Path] = field(default_factory=list)
    variant_yamls: list[Path] = field(default_factory=list)
    total_files_scanned: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "repo_root": str(self.repo_root),
            "python_files": len(self.python_files),
            "config_files": len(self.config_files),
            "doc_files": len(self.doc_files),
            "service_boundaries": len(self.service_boundaries),
            "existing_component_yamls": len(self.component_yamls),
            "existing_variant_yamls": len(self.variant_yamls),
            "total_files_scanned": self.total_files_scanned,
        }


class CodebaseScanner:
    """Scans a repository directory tree and classifies discovered files.

    The scanner walks the directory tree, skipping ignored directories,
    and classifies each file by extension and name. It also detects
    service boundaries (directories containing Dockerfiles, package.json, etc.)
    which hint at microservice architecture.
    """

    def __init__(
        self,
        ignore_patterns: set[str] | None = None,
        max_file_size_bytes: int = 5_000_000,  # 5 MB
    ) -> None:
        self._ignore = ignore_patterns or _DEFAULT_IGNORE
        self._max_file_size = max_file_size_bytes

    def scan(self, repo_root: str | Path) -> ScanResult:
        """Scan a repository and return classified file lists."""
        root = Path(repo_root).resolve()
        if not root.is_dir():
            raise ValueError(f"Repository root is not a directory: {root}")

        result = ScanResult(repo_root=root)
        service_dirs: dict[Path, str] = {}

        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)

            # Skip ignored directories
            dirnames[:] = [
                d for d in dirnames
                if d not in self._ignore
                and not any(d.endswith(pat.lstrip("*")) for pat in self._ignore if "*" in pat)
            ]

            for filename in filenames:
                filepath = current / filename
                result.total_files_scanned += 1

                try:
                    size = filepath.stat().st_size
                except OSError:
                    continue

                if size > self._max_file_size:
                    continue

                relative = str(filepath.relative_to(root))
                ext = filepath.suffix.lower()

                # Check for service markers
                if filename in _SERVICE_MARKERS:
                    service_dirs[current] = filename

                # Classify the file
                if ext in _PYTHON_EXTS:
                    df = DiscoveredFile(filepath, relative, "python", size)
                    result.python_files.append(df)
                elif ext in _CONFIG_EXTS or filename.startswith("."):
                    df = DiscoveredFile(filepath, relative, "config", size)
                    result.config_files.append(df)
                elif ext in _DOC_EXTS:
                    df = DiscoveredFile(filepath, relative, "doc", size)
                    result.doc_files.append(df)

        # Build service boundaries
        for svc_dir, marker in sorted(service_dirs.items()):
            boundary = ServiceBoundary(
                name=svc_dir.name,
                root_dir=svc_dir,
                marker_file=marker,
            )
            for pf in result.python_files:
                if pf.path.is_relative_to(svc_dir):
                    boundary.python_files.append(pf.path)
            for cf in result.config_files:
                if cf.path.is_relative_to(svc_dir):
                    boundary.config_files.append(cf.path)
            for df in result.doc_files:
                if df.path.is_relative_to(svc_dir):
                    boundary.doc_files.append(df.path)
            result.service_boundaries.append(boundary)

        # Detect existing Impact Radar YAML files
        result.component_yamls = sorted(root.glob("data/components/*.yaml"))
        result.variant_yamls = sorted(root.glob("data/variants/*.yaml"))

        return result
