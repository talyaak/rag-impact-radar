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


# ── .csproj ──────────────────────────────────────────────────────────


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _make_csproj_discovered(path: Path, repo_root: Path) -> DiscoveredFile:
    return DiscoveredFile(
        path=path,
        relative_path=str(path.relative_to(repo_root)),
        category="manifest:csproj",
        size_bytes=path.stat().st_size,
    )


def _make_sln_discovered(path: Path, repo_root: Path) -> DiscoveredFile:
    return DiscoveredFile(
        path=path,
        relative_path=str(path.relative_to(repo_root)),
        category="manifest:sln",
        size_bytes=path.stat().st_size,
    )


class TestCsprojParser:
    def test_extracts_component_from_sdk_style_csproj(self, tmp_path: Path) -> None:
        csproj = tmp_path / "src" / "PaymentsApi" / "PaymentsApi.csproj"
        _write_file(csproj, textwrap.dedent("""\
            <Project Sdk="Microsoft.NET.Sdk">
              <PropertyGroup>
                <TargetFramework>net8.0</TargetFramework>
                <AssemblyName>PaymentsApi</AssemblyName>
                <Description>Handles checkout + refunds for the storefront.</Description>
                <Authors>Payments Team</Authors>
                <PackageTags>payments;stripe;http</PackageTags>
              </PropertyGroup>
              <ItemGroup>
                <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
                <PackageReference Include="Serilog" Version="3.1.1" />
              </ItemGroup>
              <ItemGroup>
                <ProjectReference Include="..\\Payments.Core\\Payments.Core.csproj" />
              </ItemGroup>
            </Project>
        """))

        parser = CodebaseParser()
        result = parser._analyze_csproj(_make_csproj_discovered(csproj, tmp_path), tmp_path)

        assert result is not None
        component, modules = result
        assert component.name == "PaymentsApi"
        assert component.id == "paymentsapi"
        assert component.description.startswith("Handles checkout")
        assert component.team_owner == "Payments Team"
        assert component.language == "csharp"
        assert component.confidence == 1.0
        assert "payments" in component.api_surface
        assert "stripe" in component.api_surface

        module_ids = {m.id for m in modules}
        assert "newtonsoft_json" in module_ids
        assert "serilog" in module_ids
        # ProjectReference → internal module, id taken from referenced stem
        assert "payments_core" in module_ids

        by_id = {m["module_id"]: m for m in component.modules}
        assert by_id["newtonsoft_json"]["coupling"] == "tight"
        assert by_id["payments_core"]["usage"].startswith("ProjectReference")

        internal = next(m for m in modules if m.id == "payments_core")
        assert internal.description.startswith("Internal .NET project")

    def test_legacy_xmlns_csproj_parses(self, tmp_path: Path) -> None:
        csproj = tmp_path / "Legacy" / "Legacy.csproj"
        _write_file(csproj, textwrap.dedent("""\
            <Project ToolsVersion="15.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
              <PropertyGroup>
                <AssemblyName>LegacyLib</AssemblyName>
                <Description>Old .NET Framework library.</Description>
              </PropertyGroup>
              <ItemGroup>
                <PackageReference Include="log4net">
                  <Version>2.0.15</Version>
                </PackageReference>
              </ItemGroup>
            </Project>
        """))

        parser = CodebaseParser()
        component, modules = parser._analyze_csproj(
            _make_csproj_discovered(csproj, tmp_path), tmp_path
        )

        assert component.name == "LegacyLib"
        assert component.language == "csharp"
        assert {m.id for m in modules} == {"log4net"}

    def test_missing_description_yields_low_confidence(self, tmp_path: Path) -> None:
        csproj = tmp_path / "Tiny" / "Tiny.csproj"
        _write_file(csproj, textwrap.dedent("""\
            <Project Sdk="Microsoft.NET.Sdk">
              <PropertyGroup>
                <TargetFramework>net8.0</TargetFramework>
              </PropertyGroup>
            </Project>
        """))

        parser = CodebaseParser()
        component, modules = parser._analyze_csproj(
            _make_csproj_discovered(csproj, tmp_path), tmp_path
        )
        assert component.name == "Tiny"
        assert component.confidence == 0.5
        assert modules == []

    def test_malformed_csproj_returns_none(self, tmp_path: Path) -> None:
        csproj = tmp_path / "Bad" / "Bad.csproj"
        _write_file(csproj, "<Project><not closed")

        parser = CodebaseParser()
        assert parser._analyze_csproj(
            _make_csproj_discovered(csproj, tmp_path), tmp_path
        ) is None

    def test_scanner_routes_csproj_into_manifests(self, tmp_path: Path) -> None:
        _write_file(tmp_path / "A" / "A.csproj", "<Project><PropertyGroup/></Project>")
        _write_file(tmp_path / "B" / "B.csproj", "<Project><PropertyGroup/></Project>")

        scanner = CodebaseScanner()
        result = scanner.scan(tmp_path)

        assert "csproj" in result.manifests_by_kind
        assert len(result.manifests_by_kind["csproj"]) == 2


# ── .sln ─────────────────────────────────────────────────────────────


class TestSlnGrouping:
    def test_parses_project_entries_and_ignores_folders(self, tmp_path: Path) -> None:
        # Write the csproj files so path resolution lands inside the repo.
        _write_file(tmp_path / "src" / "Api" / "Api.csproj", "<Project/>")
        _write_file(tmp_path / "src" / "Core" / "Core.csproj", "<Project/>")

        sln = tmp_path / "MyApp.sln"
        sln.write_text(textwrap.dedent("""\
            Microsoft Visual Studio Solution File, Format Version 12.00
            # Visual Studio Version 17
            Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "Api", "src\\Api\\Api.csproj", "{11111111-1111-1111-1111-111111111111}"
            EndProject
            Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "Core", "src\\Core\\Core.csproj", "{22222222-2222-2222-2222-222222222222}"
            EndProject
            Project("{2150E333-8FDC-42A3-9474-1A3956D46DE8}") = "SolutionItems", "SolutionItems", "{33333333-3333-3333-3333-333333333333}"
            EndProject
            Global
            EndGlobal
        """))

        parser = CodebaseParser()
        result = parser._analyze_sln(_make_sln_discovered(sln, tmp_path), tmp_path)

        assert result is not None
        sln_name, csproj_paths = result
        assert sln_name == "MyApp"
        # Solution-folder entries (no .csproj extension) are skipped.
        assert len(csproj_paths) == 2
        assert any(p.endswith("Api.csproj") for p in csproj_paths)
        assert any(p.endswith("Core.csproj") for p in csproj_paths)

    def test_scanner_registers_sln_as_service_boundary(self, tmp_path: Path) -> None:
        _write_file(tmp_path / "svc" / "App.sln", "Microsoft Visual Studio Solution File")

        scanner = CodebaseScanner()
        result = scanner.scan(tmp_path)

        assert "sln" in result.manifests_by_kind
        # The sln directory should have been registered as a service boundary.
        assert any(b.root_dir == (tmp_path / "svc").resolve() for b in result.service_boundaries)

    def test_unreadable_sln_returns_none(self, tmp_path: Path) -> None:
        sln = tmp_path / "Ghost.sln"  # does not exist

        parser = CodebaseParser()
        fake_df = DiscoveredFile(
            path=sln,
            relative_path="Ghost.sln",
            category="manifest:sln",
            size_bytes=0,
        )
        assert parser._analyze_sln(fake_df, tmp_path) is None


# ── Dispatch through CodebaseParser.parse() ──────────────────────────


class TestParserDispatch:
    def test_parse_emits_components_for_package_json_and_csproj(self, tmp_path: Path) -> None:
        _write_package_json(tmp_path / "web" / "package.json", {
            "name": "web-ui",
            "description": "Storefront web app.",
            "dependencies": {"react": "^18.0.0"},
        })
        _write_file(tmp_path / "svc" / "Api.csproj", textwrap.dedent("""\
            <Project Sdk="Microsoft.NET.Sdk">
              <PropertyGroup>
                <AssemblyName>Api</AssemblyName>
                <Description>Backend API.</Description>
              </PropertyGroup>
              <ItemGroup>
                <PackageReference Include="Serilog" Version="3.1.1" />
              </ItemGroup>
            </Project>
        """))

        scanner = CodebaseScanner()
        scan_result = scanner.scan(tmp_path)

        parser = CodebaseParser()
        parse_result = parser.parse(scan_result)

        by_lang = {c.language for c in parse_result.components}
        assert "javascript" in by_lang  # web-ui has no tsconfig / "types"
        assert "csharp" in by_lang

        ids = {c.id for c in parse_result.components}
        assert "web_ui" in ids
        assert "api" in ids

        mod_ids = {m.id for m in parse_result.modules}
        assert "react" in mod_ids
        assert "serilog" in mod_ids

    def test_parse_skips_languages_not_in_allowlist(self, tmp_path: Path) -> None:
        _write_package_json(tmp_path / "web" / "package.json", {"name": "web"})
        _write_file(tmp_path / "svc" / "Api.csproj", "<Project><PropertyGroup/></Project>")

        scanner = CodebaseScanner()
        scan_result = scanner.scan(tmp_path)

        # Only allow csharp — package.json components should be skipped.
        parser = CodebaseParser(languages={"csharp"})
        parse_result = parser.parse(scan_result)

        langs = {c.language for c in parse_result.components}
        assert "javascript" not in langs
        assert "typescript" not in langs
        assert "csharp" in langs

    def test_parse_emits_warning_for_malformed_csproj(self, tmp_path: Path) -> None:
        _write_file(tmp_path / "Bad" / "Bad.csproj", "<Project><not closed")

        scanner = CodebaseScanner()
        scan_result = scanner.scan(tmp_path)

        parser = CodebaseParser()
        parse_result = parser.parse(scan_result)

        assert any("Bad.csproj" in w for w in parse_result.warnings)
