"""Risk report formatter — transforms AnalysisResult into readable output.

=== RAG Pipeline Learning: The Last Mile of RAG ===

A RAG pipeline has three stages: Retrieve, Augment, Generate. But there's a
hidden fourth stage: PRESENTATION. The impact analyzer produces scored impacts
and variant risks; this reporter formats them for three audiences:
  - Terminal users (Rich-formatted with color-coded severity badges)
  - API consumers (JSON-serializable dict for CI/CD and front-ends)
  - Documentation (Markdown for PR comments and wikis)

The data is computed once; only the presentation layer changes. This means you
can swap scoring algorithms or add new formats (HTML, PDF) without touching
the other side of the boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.analyzer.impact import AnalysisResult, ScoredImpact, VariantRisk

# ── Constants ──────────────────────────────────────────────────────────────
_RISK_THRESHOLDS = {"CRITICAL": 7.0, "HIGH": 4.0, "MEDIUM": 2.0}

_SEVERITY_COLORS: dict[str, str] = {
    "CRITICAL": "bold red", "HIGH": "dark_orange",
    "MEDIUM": "yellow", "LOW": "green",
}

_SEVERITY_EMOJI: dict[str, str] = {
    "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢",
}

_MAX_PATH_DISPLAY_LEN = 7


class ImpactReporter:
    """Formats AnalysisResult into terminal, dict, and Markdown output.

    === RAG Pipeline Learning: Separation of Concerns ===

    The reporter knows NOTHING about how impacts were computed. It receives a
    fully-scored AnalysisResult and focuses purely on presentation — the same
    principle behind MVC / presenter patterns.
    """

    def __init__(self, analysis_result: AnalysisResult) -> None:
        self._result = analysis_result

    # ── Public API ─────────────────────────────────────────────────────────

    def to_terminal(self) -> str:
        """Rich-formatted terminal output with color-coded severity.

        === RAG Learning: Grounded Explanations ===
        Each entry shows the structural PATH (graph evidence) alongside the
        LLM explanation, letting readers verify AI reasoning against facts.
        """
        lines: list[str] = []
        sep = "═" * 55
        lines.append(f"[bold]{sep}[/bold]")
        lines.append("[bold]  IMPACT RADAR — Change Impact Analysis Report[/bold]")
        lines.append(f"[bold]{sep}[/bold]")
        lines.append("")
        lines.append(f"[bold]📋 Change Scope:[/bold] {', '.join(self._result.changed_components)}")
        lines.append("")

        direct, indirect = self._partition_impacts()

        lines.append(self._section_header("Direct Impacts (depth 1)"))
        for imp in direct:
            lines.extend(self._fmt_impact_rich(imp))
        if not direct:
            lines.append("  [dim]No direct impacts detected.[/dim]")
        lines.append("")

        lines.append(self._section_header("Indirect Impacts (depth 2+)"))
        for imp in indirect:
            lines.extend(self._fmt_impact_rich(imp))
        if not indirect:
            lines.append("  [dim]No indirect impacts detected.[/dim]")
        lines.append("")

        lines.append(self._section_header("Affected Variants"))
        for vr in self._sorted_variants():
            lines.extend(self._fmt_variant_rich(vr))
        if not self._result.variant_risks:
            lines.append("  [dim]No variants affected.[/dim]")
        lines.append("")

        lines.append(self._section_header("Summary"))
        s = self._build_summary()
        lines.append(f"  Components changed: {s['changed']}")
        lines.append(f"  Direct impacts: {s['direct_impacts']}")
        lines.append(f"  Indirect impacts: {s['indirect_impacts']}")
        lines.append(f"  Variants affected: {s['variants_affected']} of {s['variants_total']}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict for API responses.

        === RAG Learning: Schema Stability ===
        API consumers depend on this shape. In production you'd version the
        schema; here we keep it simple but consistent.
        """
        return {
            "change_scope": list(self._result.changed_components),
            "impacts": [self._impact_to_dict(i) for i in self._sorted_impacts()],
            "variant_risks": [self._variant_to_dict(v) for v in self._sorted_variants()],
            "summary": self._build_summary(),
            "llm_used": self._result.llm_used,
        }

    def to_markdown(self) -> str:
        """Markdown-formatted report for pull requests and documentation.

        === RAG Learning: Context for Code Review ===
        Posted as a PR comment, reviewers see the blast radius BEFORE reviewing
        code — they can focus on high-risk areas flagged by the report.
        """
        lines: list[str] = []
        lines.append("# Impact Radar — Change Impact Analysis Report")
        lines.append("")
        lines.append(f"**Change Scope:** {', '.join(self._result.changed_components)}")
        lines.append("")

        direct, indirect = self._partition_impacts()

        lines.append("## Direct Impacts (depth 1)")
        lines.append("")
        for imp in direct:
            lines.extend(self._fmt_impact_md(imp))
        if not direct:
            lines.append("_No direct impacts detected._")
        lines.append("")

        lines.append("## Indirect Impacts (depth 2+)")
        lines.append("")
        for imp in indirect:
            lines.extend(self._fmt_impact_md(imp))
        if not indirect:
            lines.append("_No indirect impacts detected._")
        lines.append("")

        lines.append("## Affected Variants")
        lines.append("")
        for vr in self._sorted_variants():
            lines.extend(self._fmt_variant_md(vr))
        if not self._result.variant_risks:
            lines.append("_No variants affected._")
        lines.append("")

        s = self._build_summary()
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Components changed | {s['changed']} |")
        lines.append(f"| Direct impacts | {s['direct_impacts']} |")
        lines.append(f"| Indirect impacts | {s['indirect_impacts']} |")
        lines.append(f"| Variants affected | {s['variants_affected']} of {s['variants_total']} |")
        lines.append("")
        return "\n".join(lines)

    # ── Severity / risk helpers ────────────────────────────────────────────

    def _format_severity_badge(self, criticality: str) -> str:
        """Return a Rich-markup colored badge like ``[CRITICAL]``."""
        level = criticality.upper()
        color = _SEVERITY_COLORS.get(level, "white")
        return f"[{color}]\\[{level}][/{color}]"

    def _format_impact_path(self, path: object) -> str:
        """Format a single ImpactPath with arrows, truncating long chains."""
        chain: list[str] = getattr(path, "chain", [])
        if len(chain) <= _MAX_PATH_DISPLAY_LEN:
            return " → ".join(chain)
        return " → ".join(chain[:3]) + " → ... → " + " → ".join(chain[-3:])

    def _risk_level(self, score: float) -> str:
        """Convert numeric score to CRITICAL / HIGH / MEDIUM / LOW."""
        for level in ("CRITICAL", "HIGH", "MEDIUM"):
            if score >= _RISK_THRESHOLDS[level]:
                return level
        return "LOW"

    # ── Private formatting helpers ─────────────────────────────────────────

    def _section_header(self, title: str) -> str:
        """Rich-formatted section header with horizontal rule."""
        return f"[bold]── {title} {'─' * (55 - len(title) - 4)}[/bold]"

    def _partition_impacts(self) -> tuple[list[ScoredImpact], list[ScoredImpact]]:
        """Split scored impacts into direct (depth 1) and indirect (depth 2+)."""
        direct: list[ScoredImpact] = []
        indirect: list[ScoredImpact] = []
        for impact in self._sorted_impacts():
            (direct if impact.is_direct else indirect).append(impact)
        return direct, indirect

    def _sorted_impacts(self) -> list[ScoredImpact]:
        """All scored impacts sorted by score descending."""
        return sorted(self._result.scored_impacts.values(), key=lambda i: i.score, reverse=True)

    def _sorted_variants(self) -> list[VariantRisk]:
        """All variant risks sorted by risk_score descending."""
        return sorted(self._result.variant_risks.values(), key=lambda v: v.risk_score, reverse=True)

    def _fmt_impact_rich(self, impact: ScoredImpact) -> list[str]:
        """Format a single scored impact for Rich terminal output."""
        badge = self._format_severity_badge(impact.criticality)
        lines = [f"  {badge} {impact.component_name}  [dim]score: {impact.score:.1f}[/dim]"]
        if impact.paths:
            lines.append(f"    Path: {self._format_impact_path(impact.paths[0])}")
        if impact.shared_modules:
            mods = ", ".join(f"{m} ({impact.coupling})" for m in impact.shared_modules)
            lines.append(f"    Shared modules: {mods}")
        if impact.semantic_matches:
            best = max(impact.semantic_matches, key=lambda m: m.best_score)
            lines.append(f"    Semantic similarity: {best.best_score:.2f} ({best.component_name})")
        lines.append("")
        return lines

    def _fmt_variant_rich(self, vr: VariantRisk) -> list[str]:
        """Format a single variant risk entry for Rich terminal output."""
        risk = self._risk_level(vr.risk_score)
        color = _SEVERITY_COLORS.get(risk, "white")
        total = vr.direct_count + vr.indirect_count
        name = vr.variant_name
        if vr.tier or vr.region:
            name += f" ({', '.join(p for p in (vr.tier, vr.region) if p)})"
        lines = [
            f"  [bold]{name}[/bold]  [{color}]\\[{risk} RISK][/{color}]",
            f"    {total} components affected ({vr.direct_count} direct, {vr.indirect_count} indirect)",
        ]
        if vr.llm_explanation:
            lines.append(f"    [italic]LLM Analysis: \"{vr.llm_explanation.strip()}\"[/italic]")
        lines.append("")
        return lines

    def _fmt_impact_md(self, impact: ScoredImpact) -> list[str]:
        """Format a single scored impact as Markdown."""
        level = impact.criticality.upper()
        emoji = _SEVERITY_EMOJI.get(level, "⚪")
        lines = [f"- {emoji} **{level}** — **{impact.component_name}** (score: {impact.score:.1f})"]
        if impact.paths:
            lines.append(f"  - Path: `{self._format_impact_path(impact.paths[0])}`")
        if impact.shared_modules:
            mods = ", ".join(f"`{m}` ({impact.coupling})" for m in impact.shared_modules)
            lines.append(f"  - Shared modules: {mods}")
        return lines

    def _fmt_variant_md(self, vr: VariantRisk) -> list[str]:
        """Format a single variant risk entry as Markdown."""
        risk = self._risk_level(vr.risk_score)
        emoji = _SEVERITY_EMOJI.get(risk, "⚪")
        total = vr.direct_count + vr.indirect_count
        lines = [
            f"### {vr.variant_name} {emoji} {risk} RISK", "",
            f"- **{total}** components affected ({vr.direct_count} direct, {vr.indirect_count} indirect)",
        ]
        if vr.llm_explanation:
            lines.append(f"- LLM Analysis: _{vr.llm_explanation.strip()}_")
        lines.append("")
        return lines

    # ── Summary & serialization ────────────────────────────────────────────

    def _build_summary(self) -> dict[str, Any]:
        """Compute summary statistics from the analysis result."""
        direct = sum(1 for i in self._result.scored_impacts.values() if i.is_direct)
        indirect = len(self._result.scored_impacts) - direct
        graph_variants = len(getattr(self._result.graph_result, "affected_variants", {}))
        return {
            "changed": len(self._result.changed_components),
            "direct_impacts": direct,
            "indirect_impacts": indirect,
            "total_impacts": direct + indirect,
            "variants_affected": len(self._result.variant_risks),
            "variants_total": max(graph_variants, len(self._result.variant_risks)),
            "llm_used": self._result.llm_used,
        }

    def _impact_to_dict(self, impact: ScoredImpact) -> dict[str, Any]:
        """Serialize a ScoredImpact to a JSON-friendly dict."""
        return {
            "component_id": impact.component_id,
            "component_name": impact.component_name,
            "criticality": impact.criticality,
            "is_direct": impact.is_direct,
            "depth": impact.depth,
            "score": round(impact.score, 2),
            "coupling": impact.coupling,
            "paths": [self._format_impact_path(p) for p in impact.paths],
            "shared_modules": impact.shared_modules,
            "semantic_matches": [
                {"component_id": getattr(m, "component_id", ""), "best_score": round(m.best_score, 3)}
                for m in impact.semantic_matches
            ],
        }

    def _variant_to_dict(self, vr: VariantRisk) -> dict[str, Any]:
        """Serialize a VariantRisk to a JSON-friendly dict."""
        return {
            "variant_id": vr.variant_id,
            "variant_name": vr.variant_name,
            "risk_score": round(vr.risk_score, 2),
            "risk_level": self._risk_level(vr.risk_score),
            "tier": vr.tier,
            "region": vr.region,
            "affected_components": [
                {"component_id": si.component_id, "component_name": si.component_name,
                 "score": round(si.score, 2), "is_direct": si.is_direct}
                for si in vr.affected_components
            ],
            "direct_count": vr.direct_count,
            "indirect_count": vr.indirect_count,
            "max_criticality": vr.max_criticality,
            "llm_explanation": vr.llm_explanation,
        }
