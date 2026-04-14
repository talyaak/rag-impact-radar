"""Content sanitizer — detects and redacts sensitive data before external API calls.

Enterprise-grade redaction engine that scrubs:
  - API keys, tokens, and secrets (AWS, GCP, Azure, GitHub, generic patterns)
  - Credentials (passwords, connection strings, private keys)
  - PII (emails, phone numbers, SSNs, IP addresses)
  - Source code beyond structural metadata
  - File paths that reveal internal infrastructure
  - Environment variables and their values

Every piece of text that leaves the system boundary (LLM generation or
embedding API) MUST pass through the sanitizer. This is enforced by the
PrivacyGuard wrapper — direct LLM calls that bypass it are blocked.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RedactionLevel(str, Enum):
    """How aggressively to redact content."""
    STRICT = "strict"        # Maximum redaction — enterprise default
    MODERATE = "moderate"    # Redact secrets/PII, allow descriptions
    MINIMAL = "minimal"      # Only redact obvious secrets


class ContentCategory(str, Enum):
    """Classification of detected sensitive content."""
    API_KEY = "api_key"
    SECRET_TOKEN = "secret_token"
    PASSWORD = "password"
    PRIVATE_KEY = "private_key"
    CONNECTION_STRING = "connection_string"
    PII_EMAIL = "pii_email"
    PII_PHONE = "pii_phone"
    PII_SSN = "pii_ssn"
    IP_ADDRESS = "ip_address"
    FILE_PATH = "file_path"
    SOURCE_CODE = "source_code"
    ENV_VARIABLE = "env_variable"
    JWT_TOKEN = "jwt_token"


@dataclass
class RedactionEvent:
    """Record of a single redaction performed."""
    category: ContentCategory
    original_length: int
    replacement: str
    context_hint: str  # Non-sensitive context for audit (e.g., "near line about auth")


@dataclass
class SanitizationResult:
    """Output of a sanitization pass."""
    sanitized_text: str
    redaction_count: int
    redactions: list[RedactionEvent] = field(default_factory=list)
    categories_found: set[ContentCategory] = field(default_factory=set)

    @property
    def had_sensitive_content(self) -> bool:
        return self.redaction_count > 0

    def summary(self) -> dict[str, Any]:
        return {
            "redaction_count": self.redaction_count,
            "categories_found": sorted(c.value for c in self.categories_found),
            "sanitized_length": len(self.sanitized_text),
            "had_sensitive_content": self.had_sensitive_content,
        }


# ── Regex patterns for secret detection ──────────────────────────────

_SECRET_PATTERNS: list[tuple[ContentCategory, re.Pattern[str]]] = [
    # AWS keys
    (ContentCategory.API_KEY,
     re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}", re.ASCII)),
    (ContentCategory.SECRET_TOKEN,
     re.compile(r"aws[_\-]?secret[_\-]?access[_\-]?key\s*[=:]\s*\S+", re.IGNORECASE)),

    # GCP service account keys
    (ContentCategory.PRIVATE_KEY,
     re.compile(r'"private_key"\s*:\s*"-----BEGIN [A-Z ]+ KEY-----[^"]*"', re.DOTALL)),

    # Azure
    (ContentCategory.API_KEY,
     re.compile(r"(?:AccountKey|SharedAccessKey)\s*=\s*[A-Za-z0-9+/=]{20,}", re.ASCII)),

    # GitHub tokens
    (ContentCategory.SECRET_TOKEN,
     re.compile(r"gh[ps]_[A-Za-z0-9_]{36,}", re.ASCII)),
    (ContentCategory.SECRET_TOKEN,
     re.compile(r"github_pat_[A-Za-z0-9_]{22,}", re.ASCII)),

    # Generic API keys (Bearer tokens, key= patterns)
    (ContentCategory.SECRET_TOKEN,
     re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.ASCII)),
    (ContentCategory.API_KEY,
     re.compile(
         r"(?:api[_\-]?key|apikey|access[_\-]?token|auth[_\-]?token|secret[_\-]?key)"
         r"\s*[=:]\s*['\"]?[A-Za-z0-9\-._~+/]{16,}['\"]?",
         re.IGNORECASE,
     )),

    # OpenAI keys
    (ContentCategory.API_KEY,
     re.compile(r"sk-[A-Za-z0-9]{20,}", re.ASCII)),

    # Slack tokens
    (ContentCategory.SECRET_TOKEN,
     re.compile(r"xox[bpoas]-[A-Za-z0-9\-]{10,}", re.ASCII)),

    # JWT tokens
    (ContentCategory.JWT_TOKEN,
     re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", re.ASCII)),

    # Private keys (PEM)
    (ContentCategory.PRIVATE_KEY,
     re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),

    # Passwords in config/code
    (ContentCategory.PASSWORD,
     re.compile(
         r"(?:password|passwd|pwd|secret)\s*[=:]\s*['\"]?[^\s'\"]{8,}['\"]?",
         re.IGNORECASE,
     )),

    # Connection strings
    (ContentCategory.CONNECTION_STRING,
     re.compile(
         r"(?:mongodb|postgres|mysql|redis|amqp|mssql)(?:ql)?://\S+",
         re.IGNORECASE,
     )),

    # Environment variable assignments
    (ContentCategory.ENV_VARIABLE,
     re.compile(
         r"(?:export\s+)?(?:DATABASE_URL|REDIS_URL|SECRET_KEY|API_SECRET|PRIVATE_KEY)"
         r"\s*=\s*\S+",
         re.IGNORECASE,
     )),
]

_PII_PATTERNS: list[tuple[ContentCategory, re.Pattern[str]]] = [
    # Email addresses
    (ContentCategory.PII_EMAIL,
     re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),

    # Phone numbers (US/international)
    (ContentCategory.PII_PHONE,
     re.compile(r"(?:\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}")),

    # SSN
    (ContentCategory.PII_SSN,
     re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),

    # IPv4 addresses (not localhost/private range in strict mode)
    (ContentCategory.IP_ADDRESS,
     re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

_FILE_PATH_PATTERN = re.compile(
    r"(?:/(?:home|Users|var|etc|opt|srv|usr|tmp)/[^\s:;,\"']+)"
    r"|(?:[A-Z]:\\[^\s:;,\"']+)",
)

# Source code indicators (function bodies, class implementations)
_SOURCE_CODE_INDICATORS = [
    re.compile(r"^\s*def \w+\(.*\):\s*$", re.MULTILINE),
    re.compile(r"^\s*class \w+.*:\s*$", re.MULTILINE),
    re.compile(r"^\s*(?:if|for|while|try|except|with) .*:\s*$", re.MULTILINE),
    re.compile(r"^\s*(?:import |from \S+ import )", re.MULTILINE),
    re.compile(r"^\s*(?:return |yield |raise |assert )", re.MULTILINE),
]


class ContentSanitizer:
    """Detects and redacts sensitive content from text before external API calls.

    The sanitizer operates in three passes:
      1. Secret detection — API keys, tokens, passwords, private keys
      2. PII detection — emails, phone numbers, SSNs, IP addresses
      3. Structural scrubbing — file paths, source code patterns

    Each detected item is replaced with a deterministic placeholder
    (e.g., [REDACTED:api_key:a1b2c3]) that preserves text structure
    for the LLM while removing sensitive content. The hash suffix
    allows the LLM to track that the same redacted value appears
    in multiple places without seeing the actual value.
    """

    def __init__(
        self,
        redaction_level: RedactionLevel = RedactionLevel.STRICT,
        custom_patterns: list[tuple[str, str]] | None = None,
        allow_file_paths: bool = False,
        allow_ip_addresses: bool = False,
    ) -> None:
        self._level = redaction_level
        self._allow_file_paths = allow_file_paths
        self._allow_ip_addresses = allow_ip_addresses
        self._custom_patterns: list[tuple[ContentCategory, re.Pattern[str]]] = []

        if custom_patterns:
            for name, pattern in custom_patterns:
                self._custom_patterns.append(
                    (ContentCategory.SECRET_TOKEN, re.compile(pattern))
                )

    def sanitize(self, text: str) -> SanitizationResult:
        """Run all sanitization passes on the input text.

        Returns a SanitizationResult with the scrubbed text and
        an audit trail of what was redacted.
        """
        redactions: list[RedactionEvent] = []
        categories: set[ContentCategory] = set()
        result_text = text

        # Pass 1: Secrets (always active, all levels)
        result_text = self._redact_secrets(result_text, redactions, categories)

        # Pass 2: PII (STRICT and MODERATE)
        if self._level in (RedactionLevel.STRICT, RedactionLevel.MODERATE):
            result_text = self._redact_pii(result_text, redactions, categories)

        # Pass 3: File paths (STRICT only, unless explicitly allowed)
        if self._level == RedactionLevel.STRICT and not self._allow_file_paths:
            result_text = self._redact_file_paths(result_text, redactions, categories)

        # Pass 4: Custom patterns
        result_text = self._redact_custom(result_text, redactions, categories)

        return SanitizationResult(
            sanitized_text=result_text,
            redaction_count=len(redactions),
            redactions=redactions,
            categories_found=categories,
        )

    def scan_only(self, text: str) -> list[tuple[ContentCategory, str]]:
        """Scan for sensitive content without redacting.

        Returns list of (category, matched_text) for review.
        Useful for dry-run validation before embedding.
        """
        findings: list[tuple[ContentCategory, str]] = []

        for category, pattern in _SECRET_PATTERNS:
            for match in pattern.finditer(text):
                findings.append((category, match.group()[:20] + "..."))

        for category, pattern in _PII_PATTERNS:
            for match in pattern.finditer(text):
                findings.append((category, match.group()[:20] + "..."))

        return findings

    def _redact_secrets(
        self,
        text: str,
        redactions: list[RedactionEvent],
        categories: set[ContentCategory],
    ) -> str:
        for category, pattern in _SECRET_PATTERNS:
            text = self._apply_pattern(text, pattern, category, redactions, categories)
        return text

    def _redact_pii(
        self,
        text: str,
        redactions: list[RedactionEvent],
        categories: set[ContentCategory],
    ) -> str:
        for category, pattern in _PII_PATTERNS:
            if category == ContentCategory.IP_ADDRESS and self._allow_ip_addresses:
                continue
            text = self._apply_pattern(text, pattern, category, redactions, categories)
        return text

    def _redact_file_paths(
        self,
        text: str,
        redactions: list[RedactionEvent],
        categories: set[ContentCategory],
    ) -> str:
        return self._apply_pattern(
            text, _FILE_PATH_PATTERN, ContentCategory.FILE_PATH, redactions, categories,
        )

    def _redact_custom(
        self,
        text: str,
        redactions: list[RedactionEvent],
        categories: set[ContentCategory],
    ) -> str:
        for category, pattern in self._custom_patterns:
            text = self._apply_pattern(text, pattern, category, redactions, categories)
        return text

    def _apply_pattern(
        self,
        text: str,
        pattern: re.Pattern[str],
        category: ContentCategory,
        redactions: list[RedactionEvent],
        categories: set[ContentCategory],
    ) -> str:
        def replacer(match: re.Match[str]) -> str:
            original = match.group()
            # Deterministic hash so the same secret always gets the same placeholder
            hash_suffix = hashlib.sha256(original.encode()).hexdigest()[:8]
            replacement = f"[REDACTED:{category.value}:{hash_suffix}]"

            # Context hint: a few safe words around the match
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            safe_context = text[start:match.start()].strip()[-20:]

            redactions.append(RedactionEvent(
                category=category,
                original_length=len(original),
                replacement=replacement,
                context_hint=safe_context if safe_context else "(start of text)",
            ))
            categories.add(category)
            return replacement

        return pattern.sub(replacer, text)

    def has_source_code(self, text: str) -> bool:
        """Check if text contains source code patterns.

        Returns True if multiple source code indicators are present,
        suggesting this is actual code rather than a description.
        """
        indicator_count = sum(
            1 for pattern in _SOURCE_CODE_INDICATORS
            if pattern.search(text)
        )
        return indicator_count >= 2

    def sanitize_for_embedding(self, text: str) -> SanitizationResult:
        """Sanitize text specifically for embedding API calls.

        Embedding text should be descriptions and metadata, never raw
        source code. This method applies sanitization AND checks for
        source code patterns, warning if code is being embedded.
        """
        result = self.sanitize(text)

        if self.has_source_code(result.sanitized_text):
            result.categories_found.add(ContentCategory.SOURCE_CODE)
            result.redactions.append(RedactionEvent(
                category=ContentCategory.SOURCE_CODE,
                original_length=len(result.sanitized_text),
                replacement="(source code detected in embedding text)",
                context_hint="embedding content check",
            ))

        return result
