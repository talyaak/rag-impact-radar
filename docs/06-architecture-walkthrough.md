# Chapter 6: Architecture Walkthrough — Putting It All Together

> **Reading time**: ~20 minutes
> **Prerequisites**: Chapters 1–5
> **After this chapter**: You'll understand how every module in Impact Radar
> connects, how data flows from YAML to risk report, how the config spine
> drives behavior, and how the testing strategy lets you verify everything
> without an API key.
> **Project files**: All of `src/`, `config/`, `data/`, `tests/`

---

## The Full Pipeline in One Diagram

This is the entire system. Every box is a real Python module. Every arrow
is a real function call. No hidden magic.

```
  User Input: ["auth_engine", "billing_core"]
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │                    INDEXING (one-time)                   │
  │                                                         │
  │  data/components/*.yaml ──► builder.py ──► DependencyGraph
  │                              │                          │
  │  data/variants/*.yaml ──────┘                           │
  │                                                         │
  │  DependencyGraph ──► embedder.py ──► EmbeddingDocument  │
  │                                        │                │
  │                          vector_store.py ◄──────────────┘
  │                          (Chroma DB)                    │
  └─────────────────────────────────────────────────────────┘
       │
       │  Graph + Chroma are now populated
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │                 ANALYSIS (per request)                   │
  │                                                         │
  │  ┌─── Structural Path ──┐   ┌─── Semantic Path ───┐    │
  │  │                      │   │                      │    │
  │  │  traverser.py        │   │  retriever.py        │    │
  │  │  BFS on graph        │   │  Query Chroma        │    │
  │  │  → ImpactResult      │   │  → ComponentMatch[]  │    │
  │  │                      │   │                      │    │
  │  └──────────┬───────────┘   └──────────┬───────────┘    │
  │             │                          │                │
  │             └────────┬─────────────────┘                │
  │                      ▼                                  │
  │              impact.py                                  │
  │              Merge + Score + Aggregate                   │
  │              → AnalysisResult                            │
  │                      │                                  │
  │                      ▼ (optional)                       │
  │              llm_client.py                              │
  │              Grounded risk narrative                     │
  │              → VariantRisk.llm_explanation               │
  │                      │                                  │
  │                      ▼                                  │
  │              reporter.py                                │
  │              Terminal / JSON / Markdown                   │
  └─────────────────────────────────────────────────────────┘
       │
       ▼
  ┌──────────────┐
  │  main.py     │  FastAPI serves it all via REST
  └──────────────┘
```

Two paths converge in the middle. The structural path catches things that
are *connected*. The semantic path catches things that are *similar*. The
orchestrator merges both signals and the LLM explains the result.

---

## Data Flow Trace

Let's trace the exact data types through the system for a concrete input:
`changed_components = ["auth_engine"]`.

### Stage 1: Build the Graph (`builder.py`)

```
Input:   12 YAML files from data/components/
         8  YAML files from data/variants/

Process: Parse YAML → create component/module nodes → add edges

Output:  DependencyGraph
           .graph          → NetworkX Graph (32 nodes, 47 edges)
           ._components    → dict[str, ComponentData]  (12 entries)
           ._variants      → dict[str, VariantData]    (8 entries)
           ._component_to_variants → reverse index
```

The graph is **bipartite**: component nodes connect to module nodes, never
to each other directly. Two components are related when they share a module.

### Stage 2: Traverse the Graph (`traverser.py`)

```
Input:   DependencyGraph + ["auth_engine"]

Process: BFS from auth_engine through the bipartite graph
         component → module → component → module → ...
         Track visited nodes to handle cycles
         Record every path with depth and coupling

Output:  ImpactResult
           .changed_components  → ["auth_engine"]
           .direct_impacts      → dict of ComponentImpact (depth=1)
           .indirect_impacts    → dict of ComponentImpact (depth=2+)
           .affected_variants   → dict of VariantImpact
           .all_impact_paths    → list of ImpactPath
```

Each `ImpactPath` records the exact chain: `["auth_engine", "session_store",
"websocket_gateway"]` with depth=1, min_coupling="tight". These paths become
evidence for the LLM.

### Stage 3: Embed Components (`embedder.py` — one-time)

```
Input:   DependencyGraph (component data)

Process: For each of 12 components, create 4+ documents:
         - description doc  ("AuthEngine: Core identity service...")
         - module_usage docs ("AuthEngine uses crypto_utils for...")
         - api_surface doc  ("POST /api/v1/auth/login...")
         - summary doc      ("Component: AuthEngine, Purpose: ...")

Output:  93 EmbeddingDocument objects → stored in Chroma
         Each doc has: text (for embedding), metadata (for filtering)
```

The key insight: we embed **natural language descriptions**, not raw YAML.
The embedding model understands "password hashing" better than
`coupling: tight`.

### Stage 4: Retrieve Similar Components (`retriever.py`)

```
Input:   Query: "Core identity service handling authentication..."
         (auth_engine's own description used as query)

Process: Stage 1 — Chroma returns top 10 documents by cosine similarity
         Stage 2 — Filter by threshold, exclude auth_engine, aggregate
                   multiple docs from same component into ComponentMatch

Output:  list[ComponentMatch]
           Each has: component_id, component_name, best_score,
                     criticality, matching_docs[]
```

### Stage 5: Orchestrate (`impact.py`)

```
Input:   ImpactResult + list[ComponentMatch] + config weights

Process: 1. Score each graph impact: coupling × criticality / depth
         2. Check for overlap: component found by BOTH graph AND semantic
            → 1.25× score boost
         3. Add semantic-only discoveries (not in graph)
         4. Aggregate into VariantRisk per affected variant
         5. (Optional) Send structured prompt to LLM per variant

Output:  AnalysisResult
           .scored_impacts  → dict[str, ScoredImpact]
           .variant_risks   → dict[str, VariantRisk]
           .graph_result    → raw ImpactResult (for auditing)
           .semantic_matches → raw ComponentMatch list
           .llm_used        → bool
```

### Stage 6: Report (`reporter.py`)

```
Input:   AnalysisResult

Output:  .to_terminal()  → Rich-markup string with colored badges
         .to_dict()      → JSON-serializable dict for APIs
         .to_markdown()  → GitHub-flavored markdown for PR comments
```

---

## The Config Spine

Every tunable parameter lives in `config/model_config.yaml`. Here's which
module reads which section:

```
model_config.yaml                    Module that reads it
─────────────────────────────────    ─────────────────────────
llm:
  model: "gpt-4o"                   → llm_client.py
  temperature: 0.2                  → llm_client.py
  max_tokens: 4096                  → llm_client.py
  max_retries: 3                    → llm_client.py
  retry_base_delay: 1.0             → llm_client.py

embedding:
  model: "text-embedding-3-small"   → llm_client.py
  batch_size: 100                   → llm_client.py

vector_store:
  collection_name: "..."            → vector_store.py
  persist_directory: "./chroma_db"  → vector_store.py
  distance_metric: "cosine"         → vector_store.py

chunking:
  chunk_size: 512                   → embedder.py
  chunk_overlap: 64                 → embedder.py

retrieval:
  top_k: 10                         → retriever.py
  similarity_threshold: 0.7         → retriever.py
  rerank_top_k: 5                   → retriever.py

graph:
  max_traversal_depth: 5            → traverser.py
  include_weak_coupling: true       → traverser.py

analysis:
  severity_weights:                 → impact.py
    coupling_tight: 1.0
    coupling_loose: 0.5
    coupling_optional: 0.2
  criticality_multipliers:          → impact.py
    critical: 4.0
    high: 2.0
    medium: 1.0
    low: 0.5
```

Want to make the LLM more creative? Change `temperature`. Want graph
traversal to go deeper? Change `max_traversal_depth`. Want tighter
retrieval? Raise `similarity_threshold`. No code changes needed.

---

## Testing Strategy

The test suite is designed so that **zero tests require an API key**.

```
┌──────────────────────────────────────────────────────────┐
│  TEST LAYER          WHAT IT TESTS         API KEY?      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  test_graph.py       Graph structure       No             │
│  (26 tests)          Node/edge counts,                   │
│                      shared modules,                     │
│                      variant lookups                     │
│                                                          │
│  test_traverser.py   BFS traversal         No             │
│  (34 tests)          Direct/indirect,                    │
│                      all 4 chains,                       │
│                      cycles, depth,                      │
│                      coupling filters                    │
│                                                          │
│  test_impact.py      RAG pipeline          No             │
│  (30 tests)          Uses Chroma's free                  │
│                      built-in embeddings                 │
│                      (sentence-transformers)             │
│                                                          │
│  test_analyzer.py    Scoring + Reporter    No             │
│  (24 tests)          Score formula,                      │
│                      variant aggregation,                │
│                      output formats                      │
│                                                          │
│  test_api.py         FastAPI endpoints     No             │
│  (22 tests)          All routes, error                   │
│                      handling, validation                │
│                                                          │
└──────────────────────────────────────────────────────────┘

  Total: 136 tests, 0 API keys required
```

The trick is layered isolation:

- **Graph tests** use pure Python — no external dependencies at all.
- **RAG tests** use Chroma's built-in sentence-transformers embedding, which
  runs locally. This gives real vector similarity (not mocks) without OpenAI.
- **Analyzer tests** use the real graph but skip the LLM (`use_llm=False`).
- **API tests** use FastAPI's `TestClient`, which runs the full app lifespan
  (loading the graph) but doesn't need an LLM client.

The only time you need `OPENAI_API_KEY` is running `scripts/build_embeddings.py`
to create production-quality embeddings.

---

## The Non-Obvious Impact Example

This is the scenario the entire project was built to demonstrate.

**Input**: Change `auth_engine` + `billing_core`.

### What the graph traverser finds

```
auth_engine (changed)
  │
  ├── crypto_utils ──────► encryption_service     (depth=1, tight)
  ├── session_store ─────► websocket_gateway      (depth=1, tight)
  ├── event_bus ─────────► notification_service   (depth=1, loose)
  │                  ├───► audit_logger            (depth=1, loose)
  │                  └───► workflow_engine          (depth=1, loose)
  └── access_control ────► api_gateway             (depth=1, tight)
                     └───► workflow_engine          (already found)

billing_core (changed)
  │
  ├── payment_gateway ───► pricing_engine          (depth=1, tight)
  ├── event_bus ─────────► (already found above)
  └── data_warehouse ────► report_generator        (depth=1, tight)
                     └───► data_exporter            (depth=1, loose)

Indirect discoveries (depth 2+):
  audit_logger ──► compliance_engine ──► report_generator (already direct from billing)
  notification_service ──► i18n_framework ──► data_exporter (already found)
```

### The non-obvious risk

Look at `Enterprise EU`. It has `audit_logger` and `report_generator` as
components. Here's why it's at risk:

```
  auth_engine
       │
       │ shares event_bus (loose coupling)
       │ "publishes auth lifecycle events"
       ▼
  audit_logger
       │
       │ shares compliance_engine (tight coupling)
       │ "GDPR audit trail and compliance reporting"
       ▼
  report_generator
       │
       └── Used by Enterprise EU for GDPR compliance reports
```

A developer changing AuthEngine's event publishing format wouldn't
immediately think "this might break EU compliance reports." But the graph
catches it: AuthEngine → event_bus → AuditLogger → compliance_engine →
ReportGenerator.

### What the LLM says (grounded in evidence)

> Enterprise EU is at risk because the GDPR compliance audit trail relies on
> AuditLogger, which subscribes to authentication lifecycle events via
> event_bus (confirmed: AuthEngine → event_bus → AuditLogger, depth=1,
> coupling=loose). AuditLogger feeds compliance_engine, which drives
> ReportGenerator (confirmed: depth=2, coupling=tight). Changes to
> AuthEngine's event schema could break audit log completeness, causing
> GDPR reporting gaps. **Mitigation**: verify event schema backward
> compatibility; run a compliance report diff before and after deployment.

Every claim cites a graph path. "Confirmed" maps to structural evidence.
This is grounded generation in action.

---

## Extension Points

Impact Radar is designed to be extended without rewriting core logic:

| Want to... | Do this |
|------------|---------|
| Add a new component | Create a YAML file in `data/components/`, re-run `build_embeddings.py` |
| Add a new variant | Create a YAML file in `data/variants/` |
| Change the LLM | Edit `llm.model` in `model_config.yaml`, or swap `LLMClient` |
| Add a retrieval strategy | Add a method to `SemanticRetriever` |
| New output format | Add a method to `ImpactReporter` (e.g., `to_html()`) |
| CI/CD integration | POST to `/api/v1/analyze` in your pipeline |
| Different vector store | Replace `VectorStore` (Pinecone, pgvector, Weaviate) |
| Tune scoring | Adjust `analysis.severity_weights` in config |

The config-driven design means most changes don't require touching Python
code. And the modular architecture means the changes that DO require code
are isolated to a single file.

---

## What You've Learned

If you've read all six chapters, you now understand:

1. **Why RAG exists** — LLMs hallucinate; RAG gives them facts to explain
2. **How embeddings work** — text becomes numbers; similar meanings are close
3. **Why knowledge graphs matter** — structural relationships that text
   similarity can't capture
4. **How retrieval works** — two-stage search, hybrid (graph + semantic),
   metadata filtering
5. **How to ground LLM generation** — structured prompts, evidence chains,
   uncertainty markers
6. **How it all fits together** — the data flow, config spine, testing
   strategy, and extension points

You're not just someone who uses AI anymore. You understand how to build
systems that make AI trustworthy.

---

*That's it. You made it. Go build something.*
