#!/usr/bin/env python3
"""Run the full V2 pipeline in learning mode against a target repo.

Boots the FastAPI app in-process with ``TestClient``, sets
``LEARNING_MODE=1`` before any module reads env, and narrates every
pipeline stage end-to-end:

    startup → seed data → ingest → gap detection → compile → analyze

Default target is the Impact Radar repo itself, which is pure Python
and produces real components — the narrator has meaningful output to
talk about instead of a contrived fixture. Pass ``--repo /path`` to
run against any other repository.

Examples:

    # ~20-minute narrated tour against the project's own source.
    python scripts/learn.py

    # Quick pass against a specific repo (CI / non-TTY also works).
    python scripts/learn.py --repo /path/to/my/repo

The script exits 0 on success and non-zero if any stage errors.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# Enable LEARNING_MODE BEFORE importing the app so the lifespan sees it.
os.environ["LEARNING_MODE"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from src.api import main as api_main  # noqa: E402
from src.core.learning_narrator import narrate  # noqa: E402


def _banner(text: str) -> None:
    bar = "═" * max(len(text) + 4, 60)
    print(f"\n{bar}\n  {text}\n{bar}")


def _pause(interactive: bool, prompt: str = "Press Enter to continue…") -> None:
    """Wait for user input in TTY mode; auto-advance otherwise."""
    if not interactive:
        return
    try:
        input(prompt)
    except EOFError:
        pass


def _fail(message: str, response_body: object = None) -> None:
    print(f"\n✖ {message}")
    if response_body is not None:
        print(f"   response: {response_body}")
    sys.exit(1)


def run(repo: Path, interactive: bool, enable_compile: bool) -> None:
    _banner("Impact Radar — Learning Mode Tour")
    print(
        "The V2 pipeline will run end-to-end with narration enabled at\n"
        "every phase. Each block explains what just happened and cites\n"
        "the docs/ chapter with the long-form version.\n"
    )
    print(f"Target repo: {repo}")
    _pause(interactive)

    with TestClient(api_main.app) as client:
        # ── Health / seed data ──────────────────────────────────────
        resp = client.get("/health")
        if resp.status_code != 200:
            _fail("Health check failed", resp.text)
        print(f"\n✓ /health → {resp.json()}")

        resp = client.get("/api/v1/components")
        if resp.status_code != 200:
            _fail("Seed components unavailable", resp.text)
        seed_payload = resp.json()
        seed_count = (
            seed_payload.get("count", 0)
            if isinstance(seed_payload, dict)
            else len(seed_payload)
        )
        print(f"✓ Seed components loaded: {seed_count}")
        _pause(interactive)

        # ── Ingest ──────────────────────────────────────────────────
        _banner("Phase: Ingest")
        resp = client.post(
            "/api/v2/ingest",
            json={"repo_path": str(repo), "embed": False},
        )
        if resp.status_code != 200:
            _fail("Ingest failed", resp.text)
        body = resp.json()
        # Response shape varies; pick whichever field is present.
        components_found = (
            body.get("components_ingested")
            or body.get("summary", {}).get("components_ingested")
            or body.get("summary", {}).get("graph", {}).get("component_count")
            or 0
        )
        by_language = body.get("summary", {}).get("components_by_language") or {}
        print(f"\n✓ Ingest complete: {components_found} components")
        if by_language:
            print(f"  by language: {by_language}")
        _pause(interactive)

        # ── Gap detection ───────────────────────────────────────────
        _banner("Phase: Gap Detection")
        resp = client.post("/api/v2/gaps/detect")
        if resp.status_code != 200:
            _fail("Gap detection failed", resp.text)
        gap_body = resp.json()
        total_gaps = gap_body.get("total_gaps") or gap_body.get("gap_count") or 0
        print(f"\n✓ Gaps detected: {total_gaps}")
        _pause(interactive)

        # ── Compile ─────────────────────────────────────────────────
        _banner("Phase: Recompile (YAML export)")
        if enable_compile:
            narrate("compile.export")
            resp = client.post("/api/v2/compile", json={"backup": True})
            if resp.status_code == 200:
                compile_body = resp.json()
                print(f"\n✓ Compile result: {compile_body.get('status', 'ok')}")
            else:
                print(f"\n⚠ Compile returned {resp.status_code}: {resp.text[:200]}")
        else:
            narrate("compile.export")
            print(
                "\n⚠ Skipping /api/v2/compile — it would write to data/components/\n"
                "  and data/variants/. Pass --compile to run it (intended for\n"
                "  ingesting YOUR repo, not for touring against the seed data)."
            )
        _pause(interactive)

        # ── Analyze ─────────────────────────────────────────────────
        _banner("Phase: Analyze (RAG in action)")
        resp = client.get("/api/v1/components")
        if resp.status_code == 200:
            payload = resp.json()
            comps = (
                payload
                if isinstance(payload, list)
                else payload.get("components", [])
            )
        else:
            comps = []
        if not comps:
            print("⚠ No components available to analyze — skipping analyze phase.")
        else:
            first = comps[0].get("id")
            print(f"Changed component chosen: {first}")
            resp = client.post(
                "/api/v1/analyze",
                json={"changed_components": [first], "use_llm": False},
            )
            if resp.status_code != 200:
                _fail("Analyze failed", resp.text)
            analysis = resp.json()
            affected = (
                analysis.get("total_affected")
                or len(analysis.get("scored_impacts", []))
                or 0
            )
            print(f"\n✓ Analyze complete: {affected} downstream impacts found")

    narrate("startup", extra={"status": "tour complete"})
    _banner("Done")
    print(
        "That was the full V2 pipeline with learning-mode narration.\n"
        "Re-run with --repo /path/to/other/repo to see it pick up\n"
        "TypeScript (package.json) or C# (.csproj) manifests too."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-ingest and narrate the V2 pipeline end-to-end.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root to ingest (default: the Impact Radar repo itself).",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Don't wait for Enter between phases (useful for CI / logs).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        dest="enable_compile",
        help=(
            "Actually call /api/v2/compile. This rewrites data/components/ "
            "and data/variants/ — only enable when ingesting your own repo."
        ),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not repo.is_dir():
        print(f"✖ Not a directory: {repo}", file=sys.stderr)
        sys.exit(2)

    interactive = (not args.no_pause) and sys.stdin.isatty()
    run(repo, interactive, args.enable_compile)


if __name__ == "__main__":
    main()
