"""OpenAI API wrapper with retry logic and config-driven parameters.

=== RAG Pipeline Learning: The "G" in RAG ===

RAG stands for Retrieval-Augmented GENERATION. This module handles the
Generation part — sending prompts to an LLM with retrieved context.

The key insight: we NEVER ask the LLM to generate facts from scratch.
Instead, we:
  1. RETRIEVE relevant context (graph paths + semantic search results)
  2. AUGMENT a prompt with that context
  3. GENERATE an explanation grounded in the retrieved facts

This pattern dramatically reduces hallucination because the LLM is
explaining known facts rather than inventing them. The retry logic here
ensures reliability — LLM APIs are inherently flaky (rate limits, timeouts),
and a failed API call shouldn't crash the entire impact analysis.

=== Retry Strategy: Exponential Backoff ===

LLM APIs enforce rate limits. When you hit one, the worst thing to do is
immediately retry (thundering herd). Exponential backoff (1s, 2s, 4s...)
spreads retries over time, giving the API breathing room. The max_retries
and retry_base_delay are config-driven so you can tune them without code changes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError


class LLMClient:
    """Config-driven OpenAI client with automatic retry logic.

    Reads model parameters (model name, temperature, max_tokens) from
    config/model_config.yaml so you can tune generation behavior without
    touching code — a best practice for any LLM-powered application.
    """

    def __init__(
        self,
        config_path: str | Path = "config/model_config.yaml",
        api_key: str | None = None,
    ) -> None:
        """Initialize the LLM client from config.

        Args:
            config_path: Path to model_config.yaml.
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.

        === RAG Learning: Why Config-Driven? ===

        Hardcoding model="gpt-4o" and temperature=0.7 is a common beginner
        mistake. In production RAG systems, you iterate on these parameters
        constantly:
          - Lower temperature (0.1-0.3) for factual risk explanations
          - Higher temperature (0.7-0.9) for creative suggestions
          - Different models for different cost/quality tradeoffs
        Config files make this iteration fast and auditable.
        """
        self._config = self._load_config(config_path)
        llm_config = self._config.get("llm", {})

        self._model = llm_config.get("model", "gpt-4o")
        self._temperature = llm_config.get("temperature", 0.2)
        self._max_tokens = llm_config.get("max_tokens", 4096)
        self._timeout = llm_config.get("timeout_seconds", 30)
        self._max_retries = llm_config.get("max_retries", 3)
        self._retry_base_delay = llm_config.get("retry_base_delay", 1.0)

        self._client = OpenAI(
            api_key=api_key,
            timeout=self._timeout,
        )

    @staticmethod
    def _load_config(config_path: str | Path) -> dict[str, Any]:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a completion with automatic retry on transient failures.

        Args:
            prompt: The user message / main prompt.
            system_prompt: Optional system message for role-setting.
            temperature: Override config temperature for this call.
            max_tokens: Override config max_tokens for this call.

        Returns:
            The generated text content.

        Raises:
            APIError: After all retries exhausted on non-transient errors.

        === RAG Learning: Prompt Structure ===

        In a RAG pipeline, the prompt typically has this structure:
          System: "You are a risk analyst..."
          User: "Given these facts: [RETRIEVED CONTEXT]\\n\\nExplain: [QUESTION]"

        The retrieved context (graph paths, semantic matches) goes into the
        user message. The system message sets the LLM's role and output format.
        This separation keeps the retrieval results clearly delineated from
        the instructions.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self._call_with_retry(
            messages=messages,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
        )

    def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the OpenAI API with exponential backoff retry.

        Retries on: RateLimitError, APIConnectionError, APITimeoutError.
        Does NOT retry on: AuthenticationError, BadRequestError (these are
        caller bugs, not transient failures).
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""

            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._retry_base_delay * (2 ** attempt)
                    time.sleep(delay)
                # else: fall through to raise

            except APIError:
                # Non-transient error — don't retry
                raise

        raise last_error  # type: ignore[misc]

    def get_embedding(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """Get an embedding vector for a single text.

        Args:
            text: The text to embed.
            model: Override the embedding model from config.

        Returns:
            A list of floats (the embedding vector).

        === RAG Learning: Embeddings Are the Bridge ===

        Embeddings convert human-readable text into a numeric vector that
        captures semantic meaning. Two texts about similar topics will have
        vectors that are close together (high cosine similarity).

        This is what makes semantic search possible: instead of keyword
        matching ("find 'session'"), we match by meaning ("find components
        related to user session management" — which matches both AuthEngine
        and WebSocketGateway even if they use different words).
        """
        embed_config = self._config.get("embedding", {})
        embed_model = model or embed_config.get("model", "text-embedding-3-small")

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.embeddings.create(
                    model=embed_model,
                    input=text,
                )
                return response.data[0].embedding

            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._retry_base_delay * (2 ** attempt)
                    time.sleep(delay)

            except APIError:
                raise

        raise last_error  # type: ignore[misc]

    def get_embeddings_batch(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Get embeddings for multiple texts in a single API call.

        === RAG Learning: Batch Embedding ===

        Embedding one text at a time is wasteful — each API call has ~100ms
        overhead. Batching sends multiple texts in one call, reducing total
        latency from N * 100ms to ~100ms. The batch_size in config controls
        how many texts per call (OpenAI allows up to ~2048).

        For our 12 components with ~20 document chunks each, batching means
        1-2 API calls instead of 240.
        """
        embed_config = self._config.get("embedding", {})
        embed_model = model or embed_config.get("model", "text-embedding-3-small")
        batch_size = embed_config.get("batch_size", 100)

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            last_error: Exception | None = None
            for attempt in range(self._max_retries + 1):
                try:
                    response = self._client.embeddings.create(
                        model=embed_model,
                        input=batch,
                    )
                    batch_embeddings = [d.embedding for d in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break

                except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                    last_error = e
                    if attempt < self._max_retries:
                        delay = self._retry_base_delay * (2 ** attempt)
                        time.sleep(delay)

                except APIError:
                    raise
            else:
                raise last_error  # type: ignore[misc]

        return all_embeddings

    @property
    def model(self) -> str:
        return self._model

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)
