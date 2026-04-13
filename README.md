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

## API Endpoints

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
│   └── model_config.yaml       # All tuning parameters (LLM, embeddings, retrieval)
├── data/
│   ├── components/             # 12 component YAML definitions
│   └── variants/               # 8 product variant YAML definitions
├── src/
│   ├── core/
│   │   └── llm_client.py       # OpenAI wrapper with retry logic
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
│   └── api/
│       └── main.py             # FastAPI endpoints
├── scripts/
│   ├── build_graph.py          # Inspect the dependency graph
│   └── build_embeddings.py     # Index components into Chroma
├── tests/
│   ├── test_graph.py           # 26 graph structure tests
│   ├── test_traverser.py       # 34 BFS + cycle + chain tests
│   ├── test_impact.py          # 30 vector store + embedder + retriever tests
│   ├── test_analyzer.py        # Analyzer scoring + reporter tests
│   └── test_api.py             # FastAPI endpoint tests
├── docs/                       # Zero-to-hero RAG curriculum (6 chapters)
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

This project includes a 6-chapter educational curriculum in [`docs/`](docs/README.md) that teaches RAG from scratch. No ML background needed — just daily AI experience.

| Chapter | Topic |
|---------|-------|
| [1. RAG Fundamentals](docs/01-rag-fundamentals.md) | What RAG is and why it exists |
| [2. Embeddings](docs/02-embeddings-demystified.md) | Text-to-numbers and similarity search |
| [3. Knowledge Graphs](docs/03-knowledge-graphs.md) | Bipartite graphs and BFS traversal |
| [4. Retrieval Strategies](docs/04-retrieval-strategies.md) | Hybrid retrieval and two-stage search |
| [5. Grounded Generation](docs/05-grounded-generation.md) | Prompt engineering and hallucination prevention |
| [6. Architecture](docs/06-architecture-walkthrough.md) | Full pipeline walkthrough |

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
