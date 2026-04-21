"""Tests for multi-language ingestion (TS/JS via package.json, C# via .csproj).

Mirrors the inline-fixture pattern from tests/test_ingestion.py. Each
test constructs manifests under tmp_path using textwrap.dedent and
then invokes CodebaseScanner + CodebaseParser end-to-end.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from src.ingestion.parser import CodebaseParser, ExtractedComponent, ExtractedModule
from src.ingestion.scanner import CodebaseScanner, DiscoveredFile


# ── package.json ─────────────────────────────────────────────────────


def _write_package_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _make_discovered(path: Path, repo_root: Path) -> DiscoveredFile:
    return DiscoveredFile(
        path=path,
        relative_path=str(path.relative_to(repo_root)),
        category="manifest:package_json",
        size_bytes=path.stat().st_size,
    )


class TestPackageJsonParser:
    def test_extracts_component_from_basic_package(self, tmp_path: Path) -> None:
        pkg = tmp_path / "apps" / "web" / "package.json"
        _write_package_json(pkg, {
            "name": "web-app",
            "version": "1.0.0",
            "description": "Customer-facing web frontend.",
            "author": "Frontend Team <fe@example.com>",
            "keywords": ["react", "ssr"],
            "dependencies": {"react": "^18.2.0", "lodash": "^4.17"},
            "devDependencies": {"jest": "^29.0.0"},
        })

        parser = CodebaseParser()
        result = parser._analyze_package_json(_make_discovered(pkg, tmp_path), tmp_path)

        assert result is not None
        component, modules = result
        assert component.name == "web-app"
        assert component.description == "Customer-facing web frontend."
        assert component.team_owner == "Frontend Team"
        assert component.language == "javascript"  # no tsconfig, no types
        assert component.confidence == 1.0
        assert "react" in component.api_surface
        assert "ssr" in component.api_surface

        # Modules: 2 deps + 1 devDep
        module_ids = {m.id for m in modules}
        assert "react" in module_ids
        assert "lodash" in module_ids
        assert "jest" in module_ids

        # Component carries module refs with correct coupling
        by_id = {m["module_id"]: m for m in component.modules}
        assert by_id["react"]["coupling"] == "tight"
        assert by_id["jest"]["coupling"] == "loose"

    def test_workspace_deps_become_internal(self, tmp_path: Path) -> None:
        pkg = tmp_path / "apps" / "web" / "package.json"
        _write_package_json(pkg, {
            "name": "web-app",
            "description": "Web.",
            "dependencies": {
                "@acme/auth-ui": "workspace:*",
                "react": "^18.0.0",
                "./local-pkg": "file:../local-pkg",
            },
        })

        parser = CodebaseParser()
        component, modules = parser._analyze_package_json(
            _make_discovered(pkg, tmp_path), tmp_path
        )

        by_id = {m.id: m for m in modules}
        assert "acme__auth_ui" in by_id
        assert by_id["acme__auth_ui"].description.startswith("Internal")
        assert by_id["react"].description.startswith("External")

    def test_typescript_detection_via_sibling_tsconfig(self, tmp_path: Path) -> None:
        pkg = tmp_path / "apps" / "api" / "package.json"
        _write_package_json(pkg, {"name": "api", "description": "API.", "dependencies": {}})
        (pkg.parent / "tsconfig.json").write_text("{}")

        parser = CodebaseParser()
        component, _ = parser._analyze_package_json(
            _make_discovered(pkg, tmp_path), tmp_path
        )
        assert component.language == "typescript"

    def test_typescript_detection_via_types_field(self, tmp_path: Path) -> None:
        pkg = tmp_path / "libs" / "core" / "package.json"
        _write_package_json(pkg, {
            "name": "@acme/core",
            "description": "Core.",
            "types": "dist/index.d.ts",
        })

        parser = CodebaseParser()
        component, _ = parser._analyze_package_json(
            _make_discovered(pkg, tmp_path), tmp_path
        )
        assert component.language == "typescript"
        assert component.id == "acme__core"

    def test_missing_description_yields_low_confidence(self, tmp_path: Path) -> None:
        pkg = tmp_path / "no-desc" / "package.json"
        _write_package_json(pkg, {"name": "no-desc"})

        parser = CodebaseParser()
        component, _ = parser._analyze_package_json(
            _make_discovered(pkg, tmp_path), tmp_path
        )
        assert component.description == "no-desc npm package"
        assert component.confidence == 0.5

    def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
        pkg = tmp_path / "bad" / "package.json"
        pkg.parent.mkdir(parents=True)
        pkg.write_text("{ not valid json")

        parser = CodebaseParser()
        result = parser._analyze_package_json(_make_discovered(pkg, tmp_path), tmp_path)
        assert result is None

    def test_scanner_routes_package_json_into_manifests(self, tmp_path: Path) -> None:
        _write_package_json(tmp_path / "apps" / "web" / "package.json", {"name": "web"})
        _write_package_json(tmp_path / "apps" / "api" / "package.json", {"name": "api"})

        scanner = CodebaseScanner()
        result = scanner.scan(tmp_path)

        assert "package_json" in result.manifests_by_kind
        assert len(result.manifests_by_kind["package_json"]) == 2
        # Summary should surface the per-kind counts
        summary = result.summary()
        assert summary["manifests_by_kind"]["package_json"] == 2
