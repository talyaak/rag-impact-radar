"""Tests for the three V2 cost-control guardrails.

Covers:
  - EmbeddingCache: get/put round-trip, LRU eviction, model-keyed isolation,
    hit counting, persistence across process instances.
  - LLMClient.get_embeddings_batch: cache hits skip the API entirely.
  - GapAnalyzer batched suggestions: N gaps packed into 1 LLM call, and the
    enable_llm_suggestions=false switch disables the LLM path completely.
  - CostEstimator: projects sane embedding/gap numbers on the seed graph,
    respects include_* flags, and subtracts cache hits from the projection.
  - POST /api/v2/ingest/estimate: returns the projection dict as JSON.

All tests run without any real API keys — LLM and embedding calls are
stubbed with in-memory fakes.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.llm_client import LLMClient
from src.gap_analysis.analyzer import GapAnalyzer
from src.gap_analysis.detector import GapItem, GapReport, GapSeverity, GapType
from src.graph.builder import DependencyGraph
from src.ingestion.estimator import CostEstimator
from src.rag.embedding_cache import EmbeddingCache


# ── Paths ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
COMPONENTS_DIR = DATA_DIR / "components"
VARIANTS_DIR = DATA_DIR / "variants"


# ── EmbeddingCache ───────────────────────────────────────────────────────


def test_embedding_cache_round_trip(tmp_path: Path) -> None:
    """put() then get() returns the same vector."""
    cache = EmbeddingCache(cache_path=tmp_path / "c.json")
    vec = [0.1, 0.2, 0.3]
    cache.put("hello world", "text-embedding-3-small", vec)
    got = cache.get("hello world", "text-embedding-3-small")
    assert got == vec
    assert cache.stats.hits == 1
    assert cache.stats.misses == 0


def test_embedding_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = EmbeddingCache(cache_path=tmp_path / "c.json")
    assert cache.get("never seen", "any-model") is None
    assert cache.stats.misses == 1


def test_embedding_cache_model_keyed(tmp_path: Path) -> None:
    """Same text under different models must not collide."""
    cache = EmbeddingCache(cache_path=tmp_path / "c.json")
    cache.put("same text", "model-a", [1.0, 0.0])
    cache.put("same text", "model-b", [0.0, 1.0])
    assert cache.get("same text", "model-a") == [1.0, 0.0]
    assert cache.get("same text", "model-b") == [0.0, 1.0]


def test_embedding_cache_lru_eviction(tmp_path: Path) -> None:
    """Exceeding max_entries evicts the oldest."""
    cache = EmbeddingCache(cache_path=tmp_path / "c.json", max_entries=2)
    cache.put("a", "m", [1.0])
    cache.put("b", "m", [2.0])
    cache.put("c", "m", [3.0])  # evicts "a"
    assert cache.size == 2
    assert cache.get("a", "m") is None
    assert cache.get("b", "m") == [2.0]
    assert cache.get("c", "m") == [3.0]
    assert cache.stats.evictions == 1


def test_embedding_cache_persists(tmp_path: Path) -> None:
    """flush() writes to disk and a fresh instance can read it back."""
    path = tmp_path / "c.json"
    c1 = EmbeddingCache(cache_path=path)
    c1.put("persistent", "m", [0.5, 0.5])
    c1.flush()
    assert path.exists()

    c2 = EmbeddingCache(cache_path=path)
    assert c2.get("persistent", "m") == [0.5, 0.5]


def test_embedding_cache_disabled(tmp_path: Path) -> None:
    """enabled=False makes get/put a no-op (useful for benchmarking)."""
    cache = EmbeddingCache(cache_path=tmp_path / "c.json", enabled=False)
    cache.put("x", "m", [1.0])
    assert cache.get("x", "m") is None
    assert cache.size == 0


# ── LLMClient batch embedding with cache ─────────────────────────────────


class _FakeEmbedResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [type("D", (), {"embedding": v})() for v in vectors]


class _FakeOpenAIClient:
    """Minimal stand-in for the OpenAI SDK client."""

    def __init__(self) -> None:
        self.embed_call_count = 0
        self.embed_last_input: list[str] | None = None

        outer = self

        class _Embeddings:
            def create(self, model: str, input: list[str]) -> _FakeEmbedResponse:  # noqa: A002
                outer.embed_call_count += 1
                outer.embed_last_input = list(input)
                return _FakeEmbedResponse([[float(i)] for i in range(len(input))])

        class _Chat:
            class _Completions:
                def create(self, **_: Any) -> Any:
                    raise RuntimeError("chat should not be called in embedding tests")

            completions = _Completions()

        self.embeddings = _Embeddings()
        self.chat = _Chat()


@pytest.fixture
def llm_client_with_fake_openai(tmp_path: Path) -> LLMClient:
    """Real LLMClient with a stub OpenAI SDK client and a cache in tmp_path."""
    # Write a minimal config pointing the cache at tmp_path so tests don't
    # pollute the on-disk cache file.
    cfg_path = tmp_path / "model_config.yaml"
    cfg_path.write_text(textwrap.dedent(f"""\
        llm:
          model: "gpt-4o"
          max_retries: 0
        embedding:
          model: "text-embedding-3-small"
          batch_size: 100
          cache_enabled: true
          cache_path: "{tmp_path / 'embed_cache.json'}"
        security:
          zero_data_retention: false
    """))
    client = LLMClient(config_path=cfg_path, api_key="sk-test")
    client._client = _FakeOpenAIClient()  # type: ignore[assignment]
    return client


def test_batch_embedding_populates_cache(llm_client_with_fake_openai: LLMClient) -> None:
    texts = ["alpha", "beta", "gamma"]
    vecs = llm_client_with_fake_openai.get_embeddings_batch(texts)
    assert len(vecs) == 3
    # All three were misses — one API call dispatched.
    fake: _FakeOpenAIClient = llm_client_with_fake_openai._client  # type: ignore[assignment]
    assert fake.embed_call_count == 1


def test_batch_embedding_served_from_cache_zero_api_calls(
    llm_client_with_fake_openai: LLMClient,
) -> None:
    texts = ["alpha", "beta", "gamma"]
    llm_client_with_fake_openai.get_embeddings_batch(texts)

    fake: _FakeOpenAIClient = llm_client_with_fake_openai._client  # type: ignore[assignment]
    baseline = fake.embed_call_count

    # Second request with the same texts must be 100% cache hits.
    vecs = llm_client_with_fake_openai.get_embeddings_batch(texts)
    assert len(vecs) == 3
    assert fake.embed_call_count == baseline, "cache hits should not hit the API"


def test_batch_embedding_only_sends_misses(
    llm_client_with_fake_openai: LLMClient,
) -> None:
    """Pre-warm two of three texts; verify only the miss is sent to the API."""
    cache = llm_client_with_fake_openai.embedding_cache
    cache.put("alpha", "text-embedding-3-small", [9.0])
    cache.put("beta", "text-embedding-3-small", [9.0])

    fake: _FakeOpenAIClient = llm_client_with_fake_openai._client  # type: ignore[assignment]
    llm_client_with_fake_openai.get_embeddings_batch(["alpha", "beta", "gamma"])

    assert fake.embed_last_input == ["gamma"]


# ── GapAnalyzer: batched suggestions & disable switch ────────────────────


def _gap(n: int) -> GapItem:
    return GapItem(
        gap_type=GapType.MISSING_DESCRIPTION,
        severity=GapSeverity.HIGH,
        entity_id=f"comp_{n}",
        entity_name=f"Comp {n}",
        description=f"missing description for comp_{n}",
        question=f"What does comp_{n} do?",
    )


class _RecordingLLM:
    """Captures prompts instead of calling OpenAI."""

    def __init__(self, response: str) -> None:
        self.calls: list[str] = []
        self.response = response

    def generate(self, prompt: str, **_: Any) -> str:
        self.calls.append(prompt)
        return self.response


@pytest.fixture
def tiny_graph() -> DependencyGraph:
    g = DependencyGraph()
    g.add_component({
        "id": "a",
        "name": "A",
        "team_owner": "t",
        "criticality": "high",
        "description": "desc a",
        "module_usage": [],
    })
    return g


def test_batched_suggestions_single_llm_call(
    tiny_graph: DependencyGraph, tmp_path: Path,
) -> None:
    """With batch_size=5, ten gaps collapse into two LLM calls (5+5)."""
    cfg = tmp_path / "model_config.yaml"
    cfg.write_text(textwrap.dedent("""\
        gap_analysis:
          suggestion_batch_size: 5
          enable_llm_suggestions: true
    """))

    llm = _RecordingLLM(response=json.dumps([f"sug {i}" for i in range(5)]))
    analyzer = GapAnalyzer(tiny_graph, llm_client=llm, config_path=cfg)  # type: ignore[arg-type]

    report = GapReport(gaps=[_gap(i) for i in range(10)])
    analyzer._generate_suggestions(report)

    assert len(llm.calls) == 2
    # Every gap got a non-empty suggestion
    assert all(g.suggestion for g in report.gaps)
    # First-batch suggestions came from the JSON array
    assert report.gaps[0].suggestion == "sug 0"


def test_batched_suggestions_fallback_on_bad_json(
    tiny_graph: DependencyGraph, tmp_path: Path,
) -> None:
    """Unparseable batch response triggers per-gap fallback."""
    cfg = tmp_path / "model_config.yaml"
    cfg.write_text(textwrap.dedent("""\
        gap_analysis:
          suggestion_batch_size: 3
          enable_llm_suggestions: true
    """))

    # First response is garbage, subsequent per-gap responses are plain strings.
    class _Flaky(_RecordingLLM):
        def generate(self, prompt: str, **_: Any) -> str:
            self.calls.append(prompt)
            if len(self.calls) == 1:
                return "not-json { nope"
            return "per-gap suggestion"

    llm = _Flaky(response="")
    analyzer = GapAnalyzer(tiny_graph, llm_client=llm, config_path=cfg)  # type: ignore[arg-type]

    report = GapReport(gaps=[_gap(i) for i in range(3)])
    analyzer._generate_suggestions(report)

    # 1 batched call (failed) + 3 per-gap fallback calls
    assert len(llm.calls) == 4
    assert all(g.suggestion == "per-gap suggestion" for g in report.gaps)


def test_suggestions_disabled_skips_llm(
    tiny_graph: DependencyGraph, tmp_path: Path,
) -> None:
    """enable_llm_suggestions=false means zero LLM calls during start_session."""
    cfg = tmp_path / "model_config.yaml"
    cfg.write_text(textwrap.dedent("""\
        gap_analysis:
          enable_llm_suggestions: false
    """))

    llm = _RecordingLLM(response="should not be called")
    analyzer = GapAnalyzer(tiny_graph, llm_client=llm, config_path=cfg)  # type: ignore[arg-type]

    session = analyzer.start_session()
    assert llm.calls == []
    # Session is still built from detector output.
    assert session.gap_report is not None


# ── CostEstimator ────────────────────────────────────────────────────────


@pytest.fixture
def seed_repo(tmp_path: Path) -> Path:
    """A tiny repository that produces some components and at least one gap."""
    svc = tmp_path / "svc"
    svc.mkdir()
    (svc / "__init__.py").write_text('"""Demo service."""\n')
    (svc / "handler.py").write_text(textwrap.dedent('''\
        """Handles requests."""

        class RequestHandler:
            """Handles incoming requests."""

            def get(self, path: str) -> dict:
                return {"path": path}

            def post(self, path: str, body: dict) -> dict:
                return {"path": path, "body": body}

            def put(self, path: str, body: dict) -> dict:
                return {"path": path, "body": body}
    '''))
    return tmp_path


def _estimator_config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pricing": {
            "embedding_per_million_tokens": 0.02,
            "llm_input_per_million_tokens": 2.50,
            "llm_output_per_million_tokens": 10.00,
            "chars_per_token": 4,
        },
        "embedding": {
            "batch_size": 100,
            "model": "text-embedding-3-small",
        },
        "gap_analysis": {
            "suggestion_batch_size": 10,
            "enable_llm_suggestions": True,
        },
    }
    base.update(overrides)
    return base


def test_cost_estimator_runs_without_api(seed_repo: Path) -> None:
    est = CostEstimator(_estimator_config())
    projection = est.estimate(seed_repo)

    d = projection.to_dict()
    assert d["repo_path"] == str(seed_repo)
    assert d["components_found"] >= 0
    assert d["embeddings"]["document_count"] >= 0
    assert d["total_cost_usd"] >= 0.0


def test_cost_estimator_respects_include_flags(seed_repo: Path) -> None:
    est = CostEstimator(_estimator_config())
    p = est.estimate(
        seed_repo,
        include_embeddings=False,
        include_gap_suggestions=False,
    )
    assert p.embeddings.document_count == 0
    assert p.gap_suggestions.llm_calls == 0
    assert p.total_cost_usd == 0.0


def test_cost_estimator_cache_hits_reduce_cost(
    seed_repo: Path, tmp_path: Path,
) -> None:
    """If every doc is already cached, projected embedding cost is 0."""
    cfg = _estimator_config()
    est_cold = CostEstimator(cfg)
    cold = est_cold.estimate(seed_repo)
    if cold.embeddings.document_count == 0:
        pytest.skip("seed repo produced no embeddings — nothing to assert")

    # Pre-populate a cache that claims to already know every doc.
    cache = EmbeddingCache(cache_path=tmp_path / "c.json")

    class _AllHitCache:
        """Stand-in that reports every lookup as a hit."""

        def get(self, _text: str, _model: str) -> list[float]:
            return [0.0]

    est_warm = CostEstimator(cfg, embedding_cache=_AllHitCache())
    warm = est_warm.estimate(seed_repo)

    assert warm.embeddings.cache_hits_assumed == warm.embeddings.document_count
    assert warm.embeddings.cost_usd == 0.0
    assert warm.total_cost_usd <= cold.total_cost_usd


# ── Endpoint: POST /api/v2/ingest/estimate ───────────────────────────────


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_estimate_endpoint_returns_projection(client: TestClient, seed_repo: Path) -> None:
    resp = client.post(
        "/api/v2/ingest/estimate",
        json={"repo_path": str(seed_repo)},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Projection shape
    assert "repo_path" in body
    assert "components_found" in body
    assert "embeddings" in body
    assert "gap_suggestions" in body
    assert "total_cost_usd" in body
    assert body["repo_path"] == str(seed_repo)


def test_estimate_endpoint_respects_flags(client: TestClient, seed_repo: Path) -> None:
    resp = client.post(
        "/api/v2/ingest/estimate",
        json={
            "repo_path": str(seed_repo),
            "include_embeddings": False,
            "include_gap_suggestions": False,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["embeddings"]["document_count"] == 0
    assert body["gap_suggestions"]["llm_calls"] == 0
    assert body["total_cost_usd"] == 0.0
