# Impact Radar

**Change Impact Analyzer** — surfaces non-obvious blast radius across product variants when components change.

Given a set of changed components, Impact Radar:
1. Traverses a dependency graph to find all affected variants
2. Uses semantic search to discover conceptually related components
3. Generates a human-readable risk report via LLM (optional)

```
              ┌──────────────────────────────────────────────────┐
              │              IMPACT RADAR PIPELINE               │
              └──────────────────────────────────────────────────┘

  Input: ["auth_engine", "billing_core"]
              │
              ├──── Structural Path ────┐      ┌── Semantic Path ──┐
              │                         │      │                   │
              ▼                         ▼      ▼                   │
     ┌─────────────────┐     ┌─────────────────────┐              │
     │   YAML Seed     │     │   Chroma Vector DB   │              │
     │   Data (12      │     │   (93 embedded docs)  │             │
     │   components,   │     └──────────┬────────────┘             │
     │   8 variants)   │               │                           │
     └────────┬────────┘     Semantic retrieval                    │
              │              finds conceptual                      │
     Graph traversal         matches                               │
     finds structural                                              │
     dependencies            ┌──────────────────┐                  │
              │              │  SemanticRetriever │                 │
              ▼              │  → ComponentMatch  │                 │
     ┌─────────────────┐     │     list           │                │
     │  ImpactTraverser │     └──────────┬────────┘                │
     │  BFS → paths,   │               │                           │
     │  affected nodes  │               │                           │
     └────────┬────────┘               │                           │
              │                         │                           │
              └────────┬────────────────┘                           │
                       ▼                                            │
              ┌──────────────────┐                                  │
              │  ImpactAnalyzer   │  Merge + Score + Boost          │
              │  → AnalysisResult │                                 │
              └────────┬─────────┘                                  │
                       │                                            │
                       ▼ (optional)                                 │
              ┌──────────────────┐                                  │
              │  LLMClient       │  Grounded risk narrative         │
              │  (GPT-4o)        │                                  │
              └────────┬─────────┘                                  │
                       │                                            │
                       ▼                                            │
              ┌──────────────────┐                                  │
              │  ImpactReporter   │  Terminal / JSON / Markdown     │
              └──────────────────┘
```

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/talyaak/rag-impact-radar.git
cd rag-impact-radar
pip install -r requirements.txt

# Run tests (no API key needed)
pytest tests/ -v

# Build the vector store (requires OPENAI_API_KEY for real embeddings)
export OPENAI_API_KEY="sk-..."
python scripts/build_embeddings.py

# Start the API server
uvicorn src.api.main:app --reload

# Run an analysis
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"changed_components": ["auth_engine", "billing_core"], "use_llm": true}'
```

---

## Non-Obvious Impact Example

This is the flagship scenario — the whole reason Impact Radar exists.

**Change scope:** `auth_engine` + `billing_core`

**Obvious risks** (any engineer would spot these):
- Starter, Pro, Enterprise plans all use AuthEngine and BillingCore directly
- EncryptionService shares `crypto_utils` with AuthEngine

**Non-obvious risk** (Impact Radar catches this):

```
  auth_engine
       │
       │ shares event_bus (loose coupling)
       ▼
  audit_logger
       │
       │ shares compliance_engine (tight coupling)
       ▼
  report_generator ← used by Enterprise EU for GDPR compliance reports!
```

**Enterprise EU is at risk** — not because of a direct dependency on AuthEngine,
but because the GDPR compliance audit trail flows through AuditLogger, which
shares `event_bus` with AuthEngine. Changes to how AuthEngine publishes events
could silently break compliance reporting for EU customers.

**LLM output** (grounded in graph evidence):

> Enterprise EU is at risk because the GDPR compliance audit trail relies on
> AuditLogger, which subscribes to authentication lifecycle events via
> event_bus (confirmed: AuthEngine → event_bus → AuditLogger, depth=1,
> coupling=loose). AuditLogger feeds compliance_engine, which drives
> ReportGenerator (confirmed: depth=2, coupling=tight). Changes to
> AuthEngine's event schema or frequency could silently break audit log
> completeness, causing GDPR reporting gaps. **Mitigation**: verify event
> schema backward compatibility; run a compliance report diff before and
> after deployment.

This explanation is **grounded** — every claim cites a graph path and coupling
strength. The LLM explains; it doesn't invent.

---

## V2 Evolution — From Seeded Demo to Self-Adapting System

V1 ships with a hand-crafted dependency graph (12 components, 8 variants). V2
turns Impact Radar into a **self-adapting system** that can onboard any user
codebase, refine its own knowledge graph through an LLM-driven dialog with the
user, and recompile itself for deployment — all while enforcing enterprise
privacy guardrails on every external call.

```
         V1 (static demo)                     V2 (self-adapting product)
  ┌──────────────────────────┐        ┌───────────────────────────────────┐
  │ hand-written YAMLs       │        │ point at any repository           │
  │ 12 components, 8 variants │   →    │ scan → AST parse → extract        │
  │ analyze immediately      │        │ detect gaps → ask user → refine   │
  │                          │        │ recompile → generate tests         │
  │                          │        │ sanitize + audit every API call    │
  └──────────────────────────┘        └───────────────────────────────────┘
```

**Three new subsystems, one new privacy layer:**

| Module | Role | Entry point |
|--------|------|-------------|
| `src/ingestion/` | Universal codebase onboarding: scan files, AST-parse Python, build graph | `IngestionEngine.ingest(repo_path)` |
| `src/gap_analysis/` | Detect orphans/ambiguities and run an interactive LLM refinement loop | `GapAnalyzer.start_session()` |
| `src/recompiler/` | Persist the curated graph as YAML + embeddings + deployment manifest, and auto-generate grounded tests | `DynamicRecompiler.compile()` |
| `src/core/privacy_guard.py` | Single gateway that sanitizes, classifies, and audits every external API call (plus local-only mode and zero-training headers) | `PrivacyGuard.generate(...)` |

**Onboarding flow** (end-to-end):

```
  repo_path
      │
      ▼
  [ingestion] scan → parse (AST) → ExtractedComponents
      │
      ▼
  [graph builder] bipartite DependencyGraph (from V1, reused as-is)
      │
      ▼
  [gap_analysis] detect → LLM suggests → user answers → graph mutates
      │             ┌──────────────────────────┐
      │             │ orphans, missing descs,  │
      │             │ coupling ambiguity, etc. │
      │             └──────────────────────────┘
      ▼
  [recompiler] export YAML + embed + write manifest → production-ready
      │
      ▼
  [test_generator] auto-generate pytest cases grounded in the graph
```

Every arrow that touches an external LLM or embedding API goes through
`PrivacyGuard`, which redacts secrets/PII, enforces size and deny-list
guardrails, and appends a tamper-evident entry to `privacy_audit.jsonl`.

See [Chapter 7: The V2 Evolution](docs/07-v2-evolution.md) for the full
architectural walkthrough and the design tradeoffs behind each subsystem.

---

## API Endpoints

### V1 — static seed data

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/api/v1/components` | List all 12 components |
| GET | `/api/v1/components/{id}` | Single component with modules |
| GET | `/api/v1/variants` | List all 8 product variants |
| GET | `/api/v1/variants/{id}` | Single variant with component manifest |
| GET | `/api/v1/graph/summary` | Graph statistics |
| POST | `/api/v1/analyze` | Full impact analysis |
| POST | `/api/v1/embeddings/rebuild` | Rebuild vector store |

### V2 — onboarding, refinement, and privacy

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v2/ingest` | Scan and parse a repository into a dependency graph |
| POST | `/api/v2/ingest/yaml` | V1-compatible ingestion from existing YAML directories |
| POST | `/api/v2/gaps/detect` | Run structural gap detection on the current graph |
| POST | `/api/v2/gaps/start-session` | Begin an interactive gap-analysis session with LLM suggestions |
| GET  | `/api/v2/gaps/questions` | Get pending questions from the active session |
| POST | `/api/v2/gaps/answer` | Submit answers and refine the graph |
| POST | `/api/v2/compile` | Recompile the curated graph into deployment artifacts |
| POST | `/api/v2/generate-tests` | Generate grounded pytest cases from the current graph |
| GET  | `/api/v2/status` | Onboarding pipeline status |
| GET  | `/api/v2/privacy/audit` | Read the append-only privacy audit log |
| GET  | `/api/v2/privacy/status` | Current redaction level, local-only flag, and guardrail config |

### Analysis Request

```json
{
  "changed_components": ["auth_engine", "billing_core"],
  "use_llm": true,
  "max_depth": 5
}
```

### Analysis Response (abbreviated)

```json
{
  "changed_components": ["auth_engine", "billing_core"],
  "direct_impacts": {
    "encryption_service": {
      "criticality": "high",
      "depth": 1,
      "paths": [{"chain": ["auth_engine", "crypto_utils", "encryption_service"]}]
    }
  },
  "indirect_impacts": {
    "report_generator": {
      "criticality": "medium",
      "depth": 2,
      "paths": [{"chain": ["auth_engine", "event_bus", "audit_logger", "compliance_engine", "report_generator"]}]
    }
  },
  "affected_variants": {
    "enterprise_eu": {
      "variant_name": "Enterprise EU",
      "direct_component_count": 4,
      "indirect_component_count": 3,
      "max_criticality": "critical"
    }
  },
  "llm_narrative": "Enterprise EU is at risk because...",
  "llm_used": true
}
```

---

## Project Structure

```
impact-radar/
├── config/
│   └── model_config.yaml       # All tuning parameters (LLM, embeddings, retrieval, privacy)
├── data/
│   ├── components/             # 12 component YAML definitions (seed data)
│   └── variants/               # 8 product variant YAML definitions (seed data)
├── src/
│   ├── core/
│   │   ├── llm_client.py       # OpenAI wrapper with retry logic
│   │   ├── sanitizer.py        # (V2) Secret/PII redaction engine
│   │   └── privacy_guard.py    # (V2) Single gateway for external API calls
│   ├── graph/
│   │   ├── builder.py          # Builds NetworkX bipartite graph from YAML
│   │   └── traverser.py        # BFS blast radius with cycle detection
│   ├── rag/
│   │   ├── embedder.py         # Converts YAML → embeddable documents
│   │   ├── retriever.py        # Two-stage semantic search with aggregation
│   │   └── vector_store.py     # Chroma wrapper
│   ├── analyzer/
│   │   ├── impact.py           # Orchestrates graph + RAG + LLM
│   │   └── reporter.py         # Formats risk reports (terminal/JSON/MD)
│   ├── ingestion/              # (V2) Universal codebase ingestion
│   │   ├── scanner.py          #        Discover + classify files
│   │   ├── parser.py           #        AST extraction of components/modules
│   │   └── engine.py           #        Scan → parse → graph → embed pipeline
│   ├── gap_analysis/           # (V2) Interactive graph refinement
│   │   ├── detector.py         #        Find orphans, ambiguities, missing links
│   │   └── analyzer.py         #        LLM-driven question/answer loop
│   ├── recompiler/             # (V2) Persist curated graph for deployment
│   │   ├── recompiler.py       #        Export YAMLs + embeddings + manifest
│   │   └── test_generator.py   #        Auto-generate grounded pytest cases
│   └── api/
│       └── main.py             # FastAPI endpoints (V1 + V2)
├── scripts/
│   ├── build_graph.py          # Inspect the dependency graph
│   └── build_embeddings.py     # Index components into Chroma
├── tests/
│   ├── test_graph.py           # 26 graph structure tests
│   ├── test_traverser.py       # 34 BFS + cycle + chain tests
│   ├── test_impact.py          # 30 vector store + embedder + retriever tests
│   ├── test_analyzer.py        # Analyzer scoring + reporter tests
│   ├── test_api.py             # FastAPI V1 endpoint tests
│   ├── test_ingestion.py       # (V2) Scanner + parser + engine tests
│   ├── test_gap_analysis.py    # (V2) Detector + interactive analyzer tests
│   ├── test_recompiler.py      # (V2) Compilation + test generator tests
│   ├── test_api_v2.py          # (V2) V2 endpoint tests
│   ├── test_security.py        # (V2) Sanitizer + privacy guard tests
│   └── test_generated_impacts.py # (V2) Self-generated validation suite
├── docs/                       # Zero-to-hero RAG curriculum (7 chapters)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Seed Data

**Components** (12): AuthEngine, BillingCore, NotificationService, APIGateway, SearchService, WebSocketGateway, EncryptionService, AuditLogger, DataExporter, WorkflowEngine, PricingEngine, ReportGenerator

**Variants** (8): Starter Global, Starter EU, Professional Global, Professional EU, Enterprise Global, Enterprise EU, Enterprise APAC, Platform API

**Shared Modules** (20): crypto_utils, session_store, event_bus, access_control, payment_gateway, template_engine, cache_layer, search_index, data_warehouse, rate_limiter, config_service, i18n_framework, compliance_engine, file_storage, task_queue, logging_pipeline, feature_flags, metric_collector, webhook_dispatcher, schema_registry

**Non-obvious dependency chains** (4):
1. i18n surprise — NotificationService → i18n_framework → DataExporter
2. Billing to security — BillingCore → payment_gateway → EncryptionService
3. Audit trail cycle — AuditLogger → compliance_engine → ReportGenerator → data_warehouse → AuditLogger
4. Realtime communication trap — WebSocketGateway → session_store → AuthEngine → event_bus → NotificationService

---

## Docker

```bash
docker build -t impact-radar .
docker run -p 8000:8000 -e OPENAI_API_KEY="sk-..." impact-radar
```

---

## Learn RAG

This project includes a 7-chapter educational curriculum in [`docs/`](docs/README.md) that teaches RAG from scratch. No ML background needed — just daily AI experience.

| Chapter | Topic |
|---------|-------|
| [1. RAG Fundamentals](docs/01-rag-fundamentals.md) | What RAG is and why it exists |
| [2. Embeddings](docs/02-embeddings-demystified.md) | Text-to-numbers and similarity search |
| [3. Knowledge Graphs](docs/03-knowledge-graphs.md) | Bipartite graphs and BFS traversal |
| [4. Retrieval Strategies](docs/04-retrieval-strategies.md) | Hybrid retrieval and two-stage search |
| [5. Grounded Generation](docs/05-grounded-generation.md) | Prompt engineering and hallucination prevention |
| [6. Architecture](docs/06-architecture-walkthrough.md) | Full pipeline walkthrough |
| [7. V2 Evolution](docs/07-v2-evolution.md) | From seeded demo to self-adapting system: ingestion, gap analysis, recompilation, and the privacy layer |

---

## Tech Stack

- **Python 3.11+**
- **Chroma** — vector database for semantic search
- **OpenAI API** — embeddings (text-embedding-3-small) + generation (GPT-4o)
- **NetworkX** — dependency graph
- **FastAPI** — REST API
- **Rich** — terminal output
- **PyYAML** — config and data definitions
- **Pytest** — test suite (all tests run without API key)
