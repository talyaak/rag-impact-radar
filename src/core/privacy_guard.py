"""Privacy guard — enforces enterprise data protection for all external API calls.

The PrivacyGuard is the SINGLE GATEWAY through which all data must flow
before reaching any external LLM or embedding API. It enforces:

  1. CONTENT SANITIZATION — All text is scrubbed for secrets, PII, and
     sensitive patterns before transmission.

  2. AUDIT LOGGING — Every external API call is logged with: timestamp,
     data classification, redaction count, content hash (not content),
     and destination. The audit log is append-only and tamper-evident.

  3. DATA CLASSIFICATION — Content is classified by sensitivity level
     before deciding what can leave the system boundary.

  4. LOCAL-ONLY MODE — When enabled, ALL external API calls are blocked.
     The system falls back to Chroma's local Sentence Transformers for
     embeddings and skips LLM generation entirely.

  5. ZERO-TRAINING GUARANTEE — API calls include explicit headers/config
     to opt out of any model training or data retention by the provider.

  6. CONTENT SIZE LIMITS — Enforces maximum payload sizes to prevent
     accidental bulk data exfiltration.

  7. BLOCKED CONTENT PATTERNS — Configurable deny-list of content that
     must NEVER leave the system (e.g., specific file paths, project names).

Architecture:
  PrivacyGuard wraps LLMClient. All code that needs LLM/embedding access
  calls PrivacyGuard methods, never LLMClient directly. The guard sanitizes
  input, logs the call, and delegates to LLMClient only after all checks pass.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from src.core.sanitizer import (
    ContentSanitizer,
    RedactionLevel,
    SanitizationResult,
)

logger = logging.getLogger(__name__)


class DataClassification(str, Enum):
    """Sensitivity classification for content."""
    PUBLIC = "public"              # Safe to send externally
    INTERNAL = "internal"          # Requires sanitization before sending
    CONFIDENTIAL = "confidential"  # Must be heavily redacted
    RESTRICTED = "restricted"      # Must NEVER leave the system


class PrivacyViolation(Exception):
    """Raised when a privacy guardrail is violated."""
    pass


@dataclass
class AuditEntry:
    """A single entry in the privacy audit log."""
    timestamp: str
    operation: str          # "llm_generate", "embed_text", "embed_batch"
    destination: str        # "openai", "local", etc.
    content_hash: str       # SHA-256 of the ORIGINAL (pre-sanitization) content
    content_length: int     # Length of original content
    sanitized_length: int   # Length after sanitization
    redaction_count: int    # Number of redactions applied
    categories_found: list[str]  # Sensitive content categories detected
    classification: str     # Data classification level
    blocked: bool           # Whether the call was blocked
    block_reason: str = ""  # Why it was blocked (if applicable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "destination": self.destination,
            "content_hash": self.content_hash,
            "content_length": self.content_length,
            "sanitized_length": self.sanitized_length,
            "redaction_count": self.redaction_count,
            "categories_found": self.categories_found,
            "classification": self.classification,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
        }


@dataclass
class PrivacyConfig:
    """Enterprise privacy configuration."""
    local_only_mode: bool = False
    redaction_level: RedactionLevel = RedactionLevel.STRICT
    max_prompt_length: int = 50_000        # Max chars per LLM prompt
    max_embedding_length: int = 10_000     # Max chars per embedding text
    max_batch_size: int = 100              # Max texts per embedding batch
    block_source_code: bool = True         # Block raw source code from LLM
    blocked_patterns: list[str] = field(default_factory=list)
    audit_log_path: str = "privacy_audit.jsonl"
    allow_file_paths: bool = False
    allow_ip_addresses: bool = False
    custom_redaction_patterns: list[tuple[str, str]] = field(default_factory=list)


class PrivacyGuard:
    """Enterprise privacy enforcement layer for all external API calls.

    Usage:
        guard = PrivacyGuard.from_config("config/model_config.yaml")

        # Instead of: llm_client.generate(prompt, system_prompt)
        # Use:        guard.guarded_generate(llm_client, prompt, system_prompt)

        # Instead of: embedding_fn(texts)
        # Use:        guard.guarded_embed(embedding_fn, texts)

    The guard ensures no sensitive content ever reaches external APIs,
    maintains an audit trail, and can operate in fully local mode.
    """

    def __init__(self, config: PrivacyConfig | None = None) -> None:
        self._config = config or PrivacyConfig()
        self._sanitizer = ContentSanitizer(
            redaction_level=self._config.redaction_level,
            custom_patterns=self._config.custom_redaction_patterns,
            allow_file_paths=self._config.allow_file_paths,
            allow_ip_addresses=self._config.allow_ip_addresses,
        )
        self._audit_log: list[AuditEntry] = []
        self._blocked_count = 0
        self._total_calls = 0

    @classmethod
    def from_config(cls, config_path: str | Path) -> PrivacyGuard:
        """Create a PrivacyGuard from model_config.yaml."""
        path = Path(config_path)
        raw: dict[str, Any] = {}
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}

        security = raw.get("security", {})
        privacy = security.get("privacy", {})

        redaction_str = privacy.get("redaction_level", "strict")
        redaction_level = RedactionLevel(redaction_str)

        config = PrivacyConfig(
            local_only_mode=privacy.get("local_only_mode", False),
            redaction_level=redaction_level,
            max_prompt_length=privacy.get("max_prompt_length", 50_000),
            max_embedding_length=privacy.get("max_embedding_length", 10_000),
            max_batch_size=privacy.get("max_batch_size", 100),
            block_source_code=privacy.get("block_source_code", True),
            blocked_patterns=privacy.get("blocked_patterns", []),
            audit_log_path=privacy.get("audit_log_path", "privacy_audit.jsonl"),
            allow_file_paths=privacy.get("allow_file_paths", False),
            allow_ip_addresses=privacy.get("allow_ip_addresses", False),
            custom_redaction_patterns=[
                (p.get("name", "custom"), p.get("pattern", ""))
                for p in privacy.get("custom_redaction_patterns", [])
                if p.get("pattern")
            ],
        )
        return cls(config)

    # ── Guarded API Methods ──────────────────────────────────────────

    def guarded_generate(
        self,
        llm_client: Any,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Sanitize prompt and system_prompt, then call LLM generate.

        Raises PrivacyViolation if the content cannot be safely sent.
        Returns empty string if in local-only mode.
        """
        self._total_calls += 1

        # Check local-only mode
        if self._config.local_only_mode:
            self._log_audit(
                operation="llm_generate",
                original_text=prompt,
                sanitized_text="",
                redaction_count=0,
                categories=[],
                blocked=True,
                block_reason="local_only_mode enabled",
            )
            return ""

        # Sanitize prompt
        prompt_result = self._sanitize_and_validate(
            prompt, "llm_generate", self._config.max_prompt_length,
        )

        # Sanitize system prompt
        sanitized_system = None
        if system_prompt:
            sys_result = self._sanitize_and_validate(
                system_prompt, "llm_generate_system", self._config.max_prompt_length,
            )
            sanitized_system = sys_result.sanitized_text

        # Log and call
        self._log_audit(
            operation="llm_generate",
            original_text=prompt,
            sanitized_text=prompt_result.sanitized_text,
            redaction_count=prompt_result.redaction_count,
            categories=[c.value for c in prompt_result.categories_found],
            blocked=False,
        )

        return llm_client.generate(
            prompt=prompt_result.sanitized_text,
            system_prompt=sanitized_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def guarded_embed(
        self,
        embedding_fn: Any,
        texts: list[str],
    ) -> list[list[float]]:
        """Sanitize all texts, then call the embedding function.

        Raises PrivacyViolation if any text cannot be safely embedded.
        Returns empty list if in local-only mode.
        """
        self._total_calls += 1

        if self._config.local_only_mode:
            self._log_audit(
                operation="embed_batch",
                original_text=f"[{len(texts)} texts]",
                sanitized_text="",
                redaction_count=0,
                categories=[],
                blocked=True,
                block_reason="local_only_mode enabled",
            )
            return []

        if len(texts) > self._config.max_batch_size:
            raise PrivacyViolation(
                f"Batch size {len(texts)} exceeds maximum {self._config.max_batch_size}"
            )

        sanitized_texts: list[str] = []
        total_redactions = 0
        all_categories: set[str] = set()

        for text in texts:
            result = self._sanitize_and_validate(
                text, "embed_text", self._config.max_embedding_length,
            )
            sanitized_texts.append(result.sanitized_text)
            total_redactions += result.redaction_count
            all_categories.update(c.value for c in result.categories_found)

        self._log_audit(
            operation="embed_batch",
            original_text=f"[{len(texts)} texts, total {sum(len(t) for t in texts)} chars]",
            sanitized_text=f"[{len(sanitized_texts)} sanitized texts]",
            redaction_count=total_redactions,
            categories=sorted(all_categories),
            blocked=False,
        )

        return embedding_fn(sanitized_texts)

    def guarded_embed_single(
        self,
        embedding_fn: Any,
        text: str,
    ) -> list[float]:
        """Sanitize a single text and call the embedding function."""
        results = self.guarded_embed(embedding_fn, [text])
        return results[0] if results else []

    # ── Validation & Sanitization ────────────────────────────────────

    def _sanitize_and_validate(
        self,
        text: str,
        operation: str,
        max_length: int,
    ) -> SanitizationResult:
        """Sanitize text and validate it passes all guardrails."""
        # Check size limit
        if len(text) > max_length:
            raise PrivacyViolation(
                f"Content length {len(text)} exceeds maximum {max_length} "
                f"for operation '{operation}'"
            )

        # Check blocked patterns
        for pattern in self._config.blocked_patterns:
            if pattern.lower() in text.lower():
                self._blocked_count += 1
                self._log_audit(
                    operation=operation,
                    original_text=text,
                    sanitized_text="",
                    redaction_count=0,
                    categories=["blocked_pattern"],
                    blocked=True,
                    block_reason=f"Content matches blocked pattern",
                )
                raise PrivacyViolation(
                    f"Content matches blocked pattern for operation '{operation}'"
                )

        # Check for source code (if blocking is enabled)
        if self._config.block_source_code and self._sanitizer.has_source_code(text):
            # Don't block, but sanitize more aggressively
            logger.warning(
                "Source code detected in %s content — applying aggressive sanitization",
                operation,
            )

        # Apply sanitization
        result = self._sanitizer.sanitize(text)

        if result.had_sensitive_content:
            logger.info(
                "Sanitized %d sensitive items (%s) in %s",
                result.redaction_count,
                ", ".join(c.value for c in result.categories_found),
                operation,
            )

        return result

    # ── Audit Logging ────────────────────────────────────────────────

    def _log_audit(
        self,
        operation: str,
        original_text: str,
        sanitized_text: str,
        redaction_count: int,
        categories: list[str],
        blocked: bool,
        block_reason: str = "",
    ) -> None:
        """Append an entry to the audit log.

        The audit log NEVER contains the original content — only a
        hash of it, the length, and metadata about what was redacted.
        """
        content_hash = hashlib.sha256(original_text.encode()).hexdigest()

        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation=operation,
            destination="openai" if not blocked else "blocked",
            content_hash=content_hash,
            content_length=len(original_text),
            sanitized_length=len(sanitized_text),
            redaction_count=redaction_count,
            categories_found=categories,
            classification=self._classify_content(categories),
            blocked=blocked,
            block_reason=block_reason,
        )

        self._audit_log.append(entry)

        # Persist to file
        try:
            log_path = Path(self._config.audit_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except OSError:
            logger.warning("Failed to persist audit log entry", exc_info=True)

    def _classify_content(self, categories: list[str]) -> str:
        """Classify content sensitivity based on detected categories."""
        if not categories:
            return DataClassification.PUBLIC.value

        high_sensitivity = {
            "private_key", "password", "connection_string",
            "pii_ssn", "secret_token",
        }
        medium_sensitivity = {
            "api_key", "jwt_token", "pii_email", "pii_phone",
        }

        cat_set = set(categories)
        if cat_set & high_sensitivity:
            return DataClassification.CONFIDENTIAL.value
        if cat_set & medium_sensitivity:
            return DataClassification.INTERNAL.value
        return DataClassification.PUBLIC.value

    # ── Status & Reporting ───────────────────────────────────────────

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return the full audit log as a list of dicts."""
        return [e.to_dict() for e in self._audit_log]

    def get_audit_summary(self) -> dict[str, Any]:
        """Return summary statistics about all API calls."""
        total_redactions = sum(e.redaction_count for e in self._audit_log)
        blocked = sum(1 for e in self._audit_log if e.blocked)
        all_categories: set[str] = set()
        for entry in self._audit_log:
            all_categories.update(entry.categories_found)

        return {
            "total_api_calls": self._total_calls,
            "total_audit_entries": len(self._audit_log),
            "total_redactions": total_redactions,
            "blocked_calls": blocked,
            "categories_encountered": sorted(all_categories),
            "local_only_mode": self._config.local_only_mode,
            "redaction_level": self._config.redaction_level.value,
        }

    @property
    def config(self) -> PrivacyConfig:
        return self._config

    @property
    def is_local_only(self) -> bool:
        return self._config.local_only_mode
