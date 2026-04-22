"""Tests for src.core.learning_narrator.

Narrator is silent when LEARNING_MODE is unset or falsy; emits a
fenced block containing the topic + chapter reference when enabled;
unknown topics are a safe no-op (no crash).
"""

from __future__ import annotations

import io
from unittest import mock

import pytest

from src.core.learning_narrator import _NARRATIONS, is_enabled, narrate


class TestIsEnabled:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on"])
    def test_truthy_values_enable(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEARNING_MODE", value)
        assert is_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "bogus"])
    def test_falsy_values_disable(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEARNING_MODE", value)
        assert is_enabled() is False

    def test_unset_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEARNING_MODE", raising=False)
        assert is_enabled() is False


class TestNarrate:
    def test_silent_when_learning_mode_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEARNING_MODE", raising=False)
        buf = io.StringIO()
        narrate("startup", stream=buf)
        assert buf.getvalue() == ""

    def test_prints_topic_when_enabled(self) -> None:
        buf = io.StringIO()
        narrate("ingest.scan", enabled=True, stream=buf)
        output = buf.getvalue()
        assert "ingest.scan" in output
        assert "docs/" in output  # chapter reference is included
        # Summary text should land in the block too.
        assert _NARRATIONS["ingest.scan"]["summary"][:30] in output

    def test_extra_fields_render(self) -> None:
        buf = io.StringIO()
        narrate(
            "ingest.parse",
            enabled=True,
            stream=buf,
            extra={"components": 42, "path": "/tmp/demo"},
        )
        output = buf.getvalue()
        assert "components: 42" in output
        assert "path: /tmp/demo" in output

    def test_unknown_topic_is_safe_noop(self) -> None:
        buf = io.StringIO()
        # Should NOT raise — unknown topics silently skip.
        narrate("this.topic.does.not.exist", enabled=True, stream=buf)
        assert buf.getvalue() == ""

    def test_env_var_without_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEARNING_MODE", "1")
        buf = io.StringIO()
        narrate("startup", stream=buf)
        assert "startup" in buf.getvalue()

    def test_all_topics_have_chapter_refs(self) -> None:
        """Guardrail: every narration must cite a docs/ chapter."""
        for topic, entry in _NARRATIONS.items():
            assert entry["chapter"].startswith("docs/"), (
                f"Narration '{topic}' is missing a docs/ chapter pointer"
            )
            assert entry["summary"].strip(), f"Narration '{topic}' has empty summary"
