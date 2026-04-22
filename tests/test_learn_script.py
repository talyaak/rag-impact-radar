"""Smoke test for scripts/learn.py.

Verifies the tour script runs to completion in under N seconds against
a tiny temp repo, with --no-pause and without --compile (the default,
non-destructive mode). We don't import the script as a module because
it sets LEARNING_MODE=1 on import; instead we invoke it as a
subprocess and assert exit status + a key narration slug appears.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_learn_script_completes_on_tiny_repo(tmp_path: Path) -> None:
    # A minimal repo so ingest has SOMETHING to extract.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text(textwrap.dedent('''\
        """Example."""


        class Example:
            """Example class."""

            def a(self) -> None:
                pass

            def b(self) -> None:
                pass
    '''))

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "learn.py"),
            "--repo",
            str(tmp_path),
            "--no-pause",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, (
        f"learn.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # Narration blocks should appear across the tour.
    for slug in ("startup", "ingest.scan", "ingest.parse", "gaps.detect"):
        assert slug in result.stdout, f"Missing narration slug {slug!r}"
    # --compile is NOT set by default, so the compile-skip warning should fire.
    assert "Skipping /api/v2/compile" in result.stdout
