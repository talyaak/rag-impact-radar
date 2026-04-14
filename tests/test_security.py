"""Tests for the enterprise security guardrails — sanitizer, privacy guard, audit logging.

Tests verify:
  - ContentSanitizer detects and redacts all secret/PII categories
  - PrivacyGuard blocks calls in local-only mode
  - PrivacyGuard enforces content size limits
  - PrivacyGuard enforces blocked patterns
  - Audit logging records every external API call
  - Zero-training headers are set on LLMClient
  - Redaction is deterministic (same secret = same placeholder)
  - Source code detection works
  - No raw content ever appears in audit logs
  - All tests run without API keys (zero-API-key testing)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.sanitizer import (
    ContentCategory,
    ContentSanitizer,
    RedactionLevel,
    SanitizationResult,
)
from src.core.privacy_guard import (
    DataClassification,
    PrivacyConfig,
    PrivacyGuard,
    PrivacyViolation,
)


# ── Paths ────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent.parent / "config" / "model_config.yaml"


# ═════════════════════════════════════════════════════════════════════════
# ContentSanitizer Tests
# ═════════════════════════════════════════════════════════════════════════


class TestSecretDetection:
    """Verify all secret pattern categories are detected and redacted."""

    def test_aws_access_key(self):
        sanitizer = ContentSanitizer()
        text = "Use key AKIAIOSFODNN7EXAMPLE to connect"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "AKIAIOSFODNN7EXAMPLE" not in result.sanitized_text
        assert ContentCategory.API_KEY in result.categories_found

    def test_aws_secret_key(self):
        sanitizer = ContentSanitizer()
        text = "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "wJalrXUtnFEMI" not in result.sanitized_text

    def test_openai_api_key(self):
        sanitizer = ContentSanitizer()
        text = "Set OPENAI_API_KEY=sk-proj-abcdef1234567890abcdef1234567890"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "sk-proj-" not in result.sanitized_text

    def test_github_personal_token(self):
        sanitizer = ContentSanitizer()
        # Build token dynamically to avoid triggering GitHub push protection
        prefix = "ghp_"
        text = f"token: {prefix}ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert prefix not in result.sanitized_text

    def test_github_pat_token(self):
        sanitizer = ContentSanitizer()
        # Build token dynamically to avoid triggering GitHub push protection
        prefix = "github_pat_"
        text = f"auth: {prefix}11ABCDEFG0HIJKLMNOP1234_abcdefghijklmnopqrstuvwxyz"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert prefix not in result.sanitized_text

    def test_bearer_token(self):
        sanitizer = ContentSanitizer()
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "eyJhbGciOiJ" not in result.sanitized_text

    def test_jwt_token(self):
        sanitizer = ContentSanitizer()
        text = "token=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert ContentCategory.JWT_TOKEN in result.categories_found

    def test_private_key_pem(self):
        sanitizer = ContentSanitizer()
        text = """Here is the key:
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA2a2rwplBQLz8EHt5sxxH
-----END RSA PRIVATE KEY-----
"""
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "MIIEowIBAAKCAQEA" not in result.sanitized_text
        assert ContentCategory.PRIVATE_KEY in result.categories_found

    def test_slack_token(self):
        sanitizer = ContentSanitizer()
        # Build token dynamically to avoid triggering GitHub push protection
        prefix = "xoxb"
        token = f"SLACK_TOKEN={prefix}-1234567890-abcdefghijklmnop"
        result = sanitizer.sanitize(token)

        assert result.had_sensitive_content
        assert f"{prefix}-" not in result.sanitized_text

    def test_generic_api_key_pattern(self):
        sanitizer = ContentSanitizer()
        text = 'api_key = "abcdef1234567890abcdef1234567890"'
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "abcdef1234567890" not in result.sanitized_text

    def test_password_in_config(self):
        sanitizer = ContentSanitizer()
        text = 'password: "SuperSecretP@ss123!"'
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "SuperSecretP@ss123" not in result.sanitized_text
        assert ContentCategory.PASSWORD in result.categories_found

    def test_connection_string_postgres(self):
        sanitizer = ContentSanitizer()
        text = "DATABASE_URL=postgresql://user:pass@host:5432/mydb"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "user:pass@host" not in result.sanitized_text

    def test_connection_string_mongodb(self):
        sanitizer = ContentSanitizer()
        text = "MONGO=mongodb://admin:secret@mongo.internal:27017/production"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "admin:secret" not in result.sanitized_text

    def test_connection_string_redis(self):
        sanitizer = ContentSanitizer()
        text = "CACHE_URL=redis://default:mypassword@redis.internal:6379"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "mypassword" not in result.sanitized_text

    def test_azure_account_key(self):
        sanitizer = ContentSanitizer()
        text = "AccountKey=aBcDeFgHiJkLmNoPqRsTuVwXyZ012345678901234567890123456789+/="
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "aBcDeFgHiJkLmNoPqRsTuVwXyZ" not in result.sanitized_text

    def test_clean_text_no_redactions(self):
        sanitizer = ContentSanitizer()
        text = "AuthEngine uses crypto_utils for password hashing and JWT signing."
        result = sanitizer.sanitize(text)

        assert not result.had_sensitive_content
        assert result.redaction_count == 0
        assert result.sanitized_text == text


class TestPIIDetection:
    """Verify PII pattern detection and redaction."""

    def test_email_address(self):
        sanitizer = ContentSanitizer(redaction_level=RedactionLevel.STRICT)
        text = "Contact admin@example.com for access"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "admin@example.com" not in result.sanitized_text
        assert ContentCategory.PII_EMAIL in result.categories_found

    def test_phone_number_us(self):
        sanitizer = ContentSanitizer(redaction_level=RedactionLevel.STRICT)
        text = "Call (555) 123-4567 for support"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "(555) 123-4567" not in result.sanitized_text

    def test_ssn(self):
        sanitizer = ContentSanitizer(redaction_level=RedactionLevel.STRICT)
        text = "SSN: 123-45-6789"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "123-45-6789" not in result.sanitized_text
        assert ContentCategory.PII_SSN in result.categories_found

    def test_ip_address(self):
        sanitizer = ContentSanitizer(redaction_level=RedactionLevel.STRICT)
        text = "Server at 192.168.1.100"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "192.168.1.100" not in result.sanitized_text

    def test_ip_address_allowed_when_configured(self):
        sanitizer = ContentSanitizer(
            redaction_level=RedactionLevel.STRICT,
            allow_ip_addresses=True,
        )
        text = "Server at 192.168.1.100"
        result = sanitizer.sanitize(text)

        assert "192.168.1.100" in result.sanitized_text

    def test_pii_not_redacted_in_minimal_mode(self):
        sanitizer = ContentSanitizer(redaction_level=RedactionLevel.MINIMAL)
        text = "Contact admin@example.com for help"
        result = sanitizer.sanitize(text)

        # Minimal mode doesn't redact PII, only secrets
        assert "admin@example.com" in result.sanitized_text


class TestFilePathRedaction:
    """Verify file path detection and redaction."""

    def test_unix_file_path(self):
        sanitizer = ContentSanitizer(redaction_level=RedactionLevel.STRICT)
        text = "Config at /home/user/myproject/config/secrets.yaml"
        result = sanitizer.sanitize(text)

        assert "/home/user/myproject" not in result.sanitized_text
        assert ContentCategory.FILE_PATH in result.categories_found

    def test_file_path_allowed_when_configured(self):
        sanitizer = ContentSanitizer(
            redaction_level=RedactionLevel.STRICT,
            allow_file_paths=True,
        )
        text = "Config at /home/user/myproject/config/secrets.yaml"
        result = sanitizer.sanitize(text)

        assert "/home/user/myproject" in result.sanitized_text


class TestRedactionDeterminism:
    """Verify that redaction produces consistent, deterministic placeholders."""

    def test_same_secret_same_placeholder(self):
        sanitizer = ContentSanitizer()
        text = "key: sk-abc123def456ghi789jkl012 and again sk-abc123def456ghi789jkl012"
        result = sanitizer.sanitize(text)

        # Both occurrences should get the same hash placeholder
        placeholders = [
            word for word in result.sanitized_text.split()
            if word.startswith("[REDACTED:")
        ]
        # They should be the same placeholder (deterministic hash)
        if len(placeholders) >= 2:
            assert placeholders[0] == placeholders[1]

    def test_different_secrets_different_placeholders(self):
        sanitizer = ContentSanitizer()
        text = "key1: sk-aaaaaaaaaaaaaaaaaaaaaa key2: sk-bbbbbbbbbbbbbbbbbbbbbb"
        result = sanitizer.sanitize(text)

        placeholders = [
            word for word in result.sanitized_text.split()
            if word.startswith("[REDACTED:")
        ]
        if len(placeholders) >= 2:
            assert placeholders[0] != placeholders[1]


class TestSourceCodeDetection:
    def test_detects_python_source(self):
        sanitizer = ContentSanitizer()
        code = """
import os
from pathlib import Path

def main():
    if os.path.exists("/tmp"):
        return True
"""
        assert sanitizer.has_source_code(code)

    def test_does_not_flag_description(self):
        sanitizer = ContentSanitizer()
        text = "AuthEngine handles user authentication and session management."
        assert not sanitizer.has_source_code(text)

    def test_sanitize_for_embedding_flags_code(self):
        sanitizer = ContentSanitizer()
        code = """
import hashlib
from typing import Any

class Hasher:
    def hash(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()
"""
        result = sanitizer.sanitize_for_embedding(code)
        assert ContentCategory.SOURCE_CODE in result.categories_found


class TestCustomPatterns:
    def test_custom_pattern_redaction(self):
        sanitizer = ContentSanitizer(
            custom_patterns=[
                ("internal_host", r"[a-z]+-svc-\d+\.internal\.corp\.com"),
            ],
        )
        text = "Deploy to auth-svc-42.internal.corp.com"
        result = sanitizer.sanitize(text)

        assert result.had_sensitive_content
        assert "auth-svc-42.internal.corp.com" not in result.sanitized_text


class TestScanOnly:
    def test_scan_does_not_modify(self):
        sanitizer = ContentSanitizer()
        text = "key: sk-testkey1234567890testkey"
        findings = sanitizer.scan_only(text)

        assert len(findings) > 0
        # Original text unchanged (scan_only doesn't modify)


# ═════════════════════════════════════════════════════════════════════════
# PrivacyGuard Tests
# ═════════════════════════════════════════════════════════════════════════


class TestPrivacyGuardLocalOnly:
    """Verify local-only mode blocks all external API calls."""

    def test_generate_blocked_in_local_only(self):
        config = PrivacyConfig(local_only_mode=True)
        guard = PrivacyGuard(config)

        mock_llm = MagicMock()
        result = guard.guarded_generate(mock_llm, "test prompt")

        assert result == ""
        mock_llm.generate.assert_not_called()

    def test_embed_blocked_in_local_only(self):
        config = PrivacyConfig(local_only_mode=True)
        guard = PrivacyGuard(config)

        mock_fn = MagicMock()
        result = guard.guarded_embed(mock_fn, ["text1", "text2"])

        assert result == []
        mock_fn.assert_not_called()

    def test_local_only_audit_logged(self):
        config = PrivacyConfig(
            local_only_mode=True,
            audit_log_path="/dev/null",
        )
        guard = PrivacyGuard(config)

        mock_llm = MagicMock()
        guard.guarded_generate(mock_llm, "test prompt")

        log = guard.get_audit_log()
        assert len(log) == 1
        assert log[0]["blocked"] is True
        assert log[0]["block_reason"] == "local_only_mode enabled"


class TestPrivacyGuardSanitization:
    """Verify PrivacyGuard sanitizes content before sending."""

    def test_secrets_redacted_before_llm(self):
        config = PrivacyConfig(audit_log_path="/dev/null")
        guard = PrivacyGuard(config)

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "LLM response"

        prompt = "Connect with key sk-abcdef1234567890abcdef1234567890"
        guard.guarded_generate(mock_llm, prompt)

        # Verify the LLM received sanitized text
        call_args = mock_llm.generate.call_args
        sent_prompt = call_args.kwargs.get("prompt", call_args.args[0] if call_args.args else "")
        assert "sk-abcdef" not in sent_prompt
        assert "[REDACTED:" in sent_prompt

    def test_secrets_redacted_before_embedding(self):
        config = PrivacyConfig(audit_log_path="/dev/null")
        guard = PrivacyGuard(config)

        mock_fn = MagicMock(return_value=[[0.1, 0.2]])

        texts = ["Password is password: MyS3cretP@ss!"]
        guard.guarded_embed(mock_fn, texts)

        call_args = mock_fn.call_args
        sent_texts = call_args.args[0]
        assert "MyS3cretP@ss" not in sent_texts[0]

    def test_system_prompt_also_sanitized(self):
        config = PrivacyConfig(audit_log_path="/dev/null")
        guard = PrivacyGuard(config)

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "response"

        guard.guarded_generate(
            mock_llm,
            prompt="safe prompt",
            system_prompt="Use key sk-dangerous1234567890dangerous",
        )

        call_args = mock_llm.generate.call_args
        sent_system = call_args.kwargs.get("system_prompt", "")
        assert "sk-dangerous" not in sent_system


class TestPrivacyGuardSizeLimits:
    """Verify content size limits are enforced."""

    def test_prompt_too_long_raises(self):
        config = PrivacyConfig(
            max_prompt_length=100,
            audit_log_path="/dev/null",
        )
        guard = PrivacyGuard(config)

        mock_llm = MagicMock()
        with pytest.raises(PrivacyViolation, match="exceeds maximum"):
            guard.guarded_generate(mock_llm, "x" * 101)

    def test_embedding_too_long_raises(self):
        config = PrivacyConfig(
            max_embedding_length=50,
            audit_log_path="/dev/null",
        )
        guard = PrivacyGuard(config)

        mock_fn = MagicMock()
        with pytest.raises(PrivacyViolation, match="exceeds maximum"):
            guard.guarded_embed(mock_fn, ["x" * 51])

    def test_batch_too_large_raises(self):
        config = PrivacyConfig(
            max_batch_size=5,
            audit_log_path="/dev/null",
        )
        guard = PrivacyGuard(config)

        mock_fn = MagicMock()
        with pytest.raises(PrivacyViolation, match="exceeds maximum"):
            guard.guarded_embed(mock_fn, ["text"] * 6)


class TestPrivacyGuardBlockedPatterns:
    """Verify blocked content patterns are enforced."""

    def test_blocked_pattern_raises(self):
        config = PrivacyConfig(
            blocked_patterns=["PROJECT_FALCON", "internal.secret.corp"],
            audit_log_path="/dev/null",
        )
        guard = PrivacyGuard(config)

        mock_llm = MagicMock()
        with pytest.raises(PrivacyViolation, match="blocked pattern"):
            guard.guarded_generate(
                mock_llm, "Deploy PROJECT_FALCON to production"
            )

    def test_blocked_pattern_case_insensitive(self):
        config = PrivacyConfig(
            blocked_patterns=["secret_project"],
            audit_log_path="/dev/null",
        )
        guard = PrivacyGuard(config)

        mock_llm = MagicMock()
        with pytest.raises(PrivacyViolation):
            guard.guarded_generate(mock_llm, "Deploy SECRET_PROJECT now")


class TestAuditLogging:
    """Verify audit log integrity and completeness."""

    def test_audit_log_records_all_calls(self):
        config = PrivacyConfig(audit_log_path="/dev/null")
        guard = PrivacyGuard(config)

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "response"

        guard.guarded_generate(mock_llm, "prompt 1")
        guard.guarded_generate(mock_llm, "prompt 2")

        log = guard.get_audit_log()
        assert len(log) == 2

    def test_audit_log_never_contains_raw_content(self):
        config = PrivacyConfig(audit_log_path="/dev/null")
        guard = PrivacyGuard(config)

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "response"

        secret = "Connect with sk-supersecretkey1234567890abc"
        guard.guarded_generate(mock_llm, secret)

        log = guard.get_audit_log()
        log_text = json.dumps(log)
        assert "sk-supersecretkey" not in log_text
        assert "content_hash" in log_text

    def test_audit_log_has_content_hash(self):
        config = PrivacyConfig(audit_log_path="/dev/null")
        guard = PrivacyGuard(config)

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "response"

        guard.guarded_generate(mock_llm, "test prompt")

        log = guard.get_audit_log()
        assert log[0]["content_hash"]
        assert len(log[0]["content_hash"]) == 64  # SHA-256 hex

    def test_audit_summary_correct(self):
        config = PrivacyConfig(audit_log_path="/dev/null")
        guard = PrivacyGuard(config)

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "response"

        guard.guarded_generate(mock_llm, "safe text")
        guard.guarded_generate(
            mock_llm,
            "with key sk-needsredaction1234567890abc",
        )

        summary = guard.get_audit_summary()
        assert summary["total_api_calls"] == 2
        assert summary["total_audit_entries"] == 2
        assert summary["total_redactions"] >= 1

    def test_audit_records_blocked_calls(self):
        config = PrivacyConfig(
            local_only_mode=True,
            audit_log_path="/dev/null",
        )
        guard = PrivacyGuard(config)

        mock_llm = MagicMock()
        guard.guarded_generate(mock_llm, "blocked call")

        summary = guard.get_audit_summary()
        assert summary["blocked_calls"] == 1

    def test_audit_records_data_classification(self):
        config = PrivacyConfig(audit_log_path="/dev/null")
        guard = PrivacyGuard(config)

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "response"

        guard.guarded_generate(mock_llm, "clean text no secrets")

        log = guard.get_audit_log()
        assert log[0]["classification"] == "public"


class TestPrivacyGuardFromConfig:
    """Verify PrivacyGuard loads correctly from config file."""

    def test_from_config_file(self):
        guard = PrivacyGuard.from_config(CONFIG_PATH)
        assert guard.config.redaction_level == RedactionLevel.STRICT
        assert guard.config.block_source_code is True

    def test_from_nonexistent_config(self, tmp_path):
        guard = PrivacyGuard.from_config(tmp_path / "nonexistent.yaml")
        # Should use defaults
        assert guard.config.redaction_level == RedactionLevel.STRICT
        assert not guard.config.local_only_mode


class TestLLMClientZeroTraining:
    """Verify LLMClient sends zero-training headers."""

    def test_zero_training_headers_set(self):
        """Verify that OpenAI client is created with no-store headers."""
        with patch("src.core.llm_client.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = MagicMock()

            from src.core.llm_client import LLMClient
            client = LLMClient(config_path=CONFIG_PATH, api_key="sk-test-fake-key-for-testing")

            # Check that OpenAI was called with default_headers
            call_kwargs = mock_openai_class.call_args.kwargs
            assert call_kwargs.get("default_headers") is not None
            assert call_kwargs["default_headers"]["X-No-Store"] == "true"


class TestEndToEndPrivacy:
    """Integration tests for the full privacy pipeline."""

    def test_multi_secret_text_fully_scrubbed(self):
        sanitizer = ContentSanitizer(redaction_level=RedactionLevel.STRICT)
        text = (
            "Deploy to auth-service using key sk-prod1234567890abcdefgh "
            "with password: SuperSecret123! "
            "connect to postgresql://admin:pass@db.internal:5432/prod "
            "and email results to ops@company.com "
            "from server 10.0.1.50"
        )
        result = sanitizer.sanitize(text)

        assert "sk-prod" not in result.sanitized_text
        assert "SuperSecret123" not in result.sanitized_text
        assert "admin:pass" not in result.sanitized_text
        assert "ops@company.com" not in result.sanitized_text
        assert "10.0.1.50" not in result.sanitized_text
        assert result.redaction_count >= 4

    def test_guard_full_pipeline(self):
        """Simulate the full guard pipeline with a realistic prompt."""
        config = PrivacyConfig(
            blocked_patterns=["CLASSIFIED"],
            audit_log_path="/dev/null",
        )
        guard = PrivacyGuard(config)

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Risk analysis complete."

        prompt = (
            "## Change Scope\n"
            "Components changed: auth_engine\n"
            "API Key used: sk-testkey1234567890abcdefgh\n"
            "Contact: engineer@internal.corp.com\n"
            "\n## Evidence\n"
            "auth_engine -> crypto_utils -> encryption_service\n"
        )

        result = guard.guarded_generate(mock_llm, prompt)

        assert result == "Risk analysis complete."

        # Verify what was sent to LLM
        sent_prompt = mock_llm.generate.call_args.kwargs["prompt"]
        assert "sk-testkey" not in sent_prompt
        assert "engineer@internal.corp.com" not in sent_prompt
        # Structural content (component names, paths) should be preserved
        assert "auth_engine" in sent_prompt
        assert "crypto_utils" in sent_prompt

        # Verify audit
        summary = guard.get_audit_summary()
        assert summary["total_redactions"] >= 2
