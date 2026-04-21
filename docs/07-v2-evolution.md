# Chapter 7: Impact Radar V2 — The Evolution

V1 of Impact Radar works with hand-curated YAML seed data. You define your components, modules, and variants manually, and the system analyzes change impact against that static graph.

V2 removes that constraint. It can ingest any codebase, ask you what it doesn't understand, recompile itself around your data, and protect your proprietary code while doing it.

This chapter covers the four pillars of V2: **Ingestion**, **Gap Analysis**, **Recompilation**, and **Enterprise Security**.

---

## The V2 Onboarding Flow

```
  YOUR CODEBASE                    IMPACT RADAR V2
  ─────────────                    ───────────────
  ┌──────────────┐    POST /ingest    ┌─────────────────────┐
  │  Repository   │ ───────────────► │  1. INGESTION        │
  │  (any lang)   │                  │     Scan → Parse →   │
  └──────────────┘                  │     Graph + Vectors  │
                                     └──────────┬──────────┘
                                                │
                                     ┌──────────▼──────────┐
                                     │  2. GAP ANALYSIS     │
                                     │     Detect gaps →    │
                                     │     LLM suggests →   │
                    ◄──── Questions ──│     Ask user →       │
                    ──── Answers ───► │     Refine graph     │
                                     └──────────┬──────────┘
                                                │
                                     ┌──────────▼──────────┐
                                     │  3. RECOMPILATION    │
                                     │     Export YAML →    │
                                     │     Persist vectors →│
                                     │     Generate tests   │
                                     └──────────┬──────────┘
                                                │
                                     ┌──────────▼──────────┐
                                     │  PRODUCTION-READY    │
                                     │  Impact Radar tuned  │
                                     │  to YOUR codebase    │
                                     └─────────────────────┘
```

---

## Pillar 1: Universal Codebase Ingestion

The ingestion engine turns any repository into a populated dependency graph and vector store. It runs in three stages:

### Stage 1: Scanning (`src/ingestion/scanner.py`)

The `CodebaseScanner` walks a repository root and classifies every file:

| Category | Extensions / Markers |
|----------|---------------------|
| Python source | `.py` |
| Config files | `.yaml`, `.yml`, `.toml`, `.json`, `.ini`, `.cfg`, `.env` |
| Documentation | `.md`, `.rst`, `.txt` |
| Service boundaries | `Dockerfile`, `docker-compose.yml`, `package.json`, `go.mod`, `Cargo.toml`, `pom.xml`, etc. |

Files matching `ignore_patterns` from `model_config.yaml` (`.git`, `__pycache__`, `node_modules`, etc.) are skipped. Files larger than `max_file_size_bytes` (default 5MB) are also skipped.

The scanner produces a `ScanResult` containing categorized file lists and detected service boundaries.

### Stage 2: Parsing (`src/ingestion/parser.py`)

The `CodebaseParser` uses Python's AST module to extract structure from source files:

- **Classes** with 2+ methods become candidate components
- **Import statements** become candidate module dependencies
- **Docstrings and comments** become description text for embeddings
- **Function signatures** become API surface definitions

Each extracted item becomes an `ExtractedComponent` with a confidence score indicating how certain the parser is that it represents a real architectural component.

### Stage 3: Graph + Embedding (`src/ingestion/engine.py`)

The `IngestionEngine` orchestrates the full pipeline:

1. Run the scanner on the repository
2. Run the parser on discovered files
3. Build the bipartite graph (component ↔ module edges) from extracted data
4. Embed component descriptions into the Chroma vector store

The output is an `IngestionResult` containing the scan results, parse results, populated graph, and embedding counts.

### API

```bash
# Ingest from a live repository
curl -X POST http://localhost:8000/api/v2/ingest \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/path/to/your/repo"}'

# Or ingest from existing YAML directories (V1-compatible)
curl -X POST http://localhost:8000/api/v2/ingest/yaml \
  -H "Content-Type: application/json" \
  -d '{"components_dir": "data/components", "variants_dir": "data/variants"}'
```

### Config (`model_config.yaml`)

```yaml
ingestion:
  component_detection: "auto"    # auto | directory | class
  min_class_methods: 2           # Minimum methods for a class to be a component
  max_file_size_bytes: 5000000   # Skip files larger than 5MB
  ignore_patterns:               # Directories to skip
    - ".git"
    - "__pycache__"
    - "node_modules"
```

---

## Pillar 2: Interactive Gap Analysis

Automated parsing rarely produces a perfect graph. The gap analysis system detects what's missing and helps you fill it in.

### Gap Detection (`src/gap_analysis/detector.py`)

The `GapDetector` examines the graph for structural problems:

| Gap Type | Severity | Description |
|----------|----------|-------------|
| `orphan_component` | Critical | Component with no module connections |
| `isolated_module` | High | Module used by only one component |
| `missing_description` | Medium | Empty or auto-generated description |
| `coupling_ambiguity` | Medium | All edges defaulting to "tight" |
| `no_api_surface` | Low | Component with no documented API |
| `no_variant_assignment` | Medium | Component not in any variant |
| `potential_missing_link` | High | Components in same directory but not connected |
| `missing_criticality` | Medium | Component without a criticality level |

Each gap becomes a `GapItem` with a severity, description, and a question to ask the user.

### LLM-Driven Suggestions (`src/gap_analysis/analyzer.py`)

The `GapAnalyzer` takes the detected gaps and uses the LLM to suggest resolutions:

1. Build a context summary of the current graph state
2. For each unresolved gap, prompt the LLM with the gap description and graph context
3. The LLM suggests specific module names, coupling levels, or descriptions
4. Low-severity gaps can be auto-resolved with reasonable defaults

### Interactive Loop

The gap analysis runs as a session with iterative refinement:

```
Iteration 0:  Detect 15 gaps → LLM suggests → Present 10 questions
                                                      │
User answers 10 questions ◄───────────────────────────┘
                                                      │
Iteration 1:  Apply answers → Re-detect → 5 gaps remain → 5 questions
                                                      │
User answers 5 questions ◄────────────────────────────┘
                                                      │
Iteration 2:  Apply answers → Re-detect → Completeness ≥ 0.8 → Done
```

The session ends when the completeness score reaches the `completeness_threshold` (default 0.8).

### API

```bash
# Detect gaps (one-shot, no session)
curl http://localhost:8000/api/v2/gaps/detect

# Start an interactive session
curl -X POST http://localhost:8000/api/v2/gaps/start-session

# Get pending questions
curl http://localhost:8000/api/v2/gaps/questions

# Submit answers
curl -X POST http://localhost:8000/api/v2/gaps/answer \
  -H "Content-Type: application/json" \
  -d '{
    "answers": [
      {"gap_index": 0, "answer": "crypto_utils, session_store"},
      {"gap_index": 1, "accept_suggestion": true}
    ]
  }'
```

### Config (`model_config.yaml`)

```yaml
gap_analysis:
  max_questions_per_round: 10
  auto_resolve_low_severity: true
  completeness_threshold: 0.8
```

---

## Pillar 3: Recompilation & Test Generation

Once the graph is refined, the recompiler persists everything into a deployment-ready state.

### Recompilation (`src/recompiler/recompiler.py`)

The `DynamicRecompiler`:

1. **Exports** the curated graph as component/variant YAML files
2. **Persists** embeddings in the Chroma vector store
3. **Updates** `model_config.yaml` with tuned parameters (e.g., adjusted traversal depth)
4. **Generates** a deployment manifest with timestamps and compilation metadata
5. **Validates** the compiled state is production-ready
6. Optionally **backs up** existing data before overwriting

### Test Generation (`src/recompiler/test_generator.py`)

The `DynamicTestGenerator` creates pytest cases grounded in the actual graph:

| Test Category | What It Validates |
|--------------|-------------------|
| `direct_impact` | Depth-1 connections via shared modules |
| `indirect_impact` | Multi-hop paths through the graph |
| `variant` | Components correctly assigned to variants |
| `embedding` | Components are embedded and retrievable |
| `cycle` | Known cycles are handled without infinite loops |

Every generated test cites its grounding evidence — the specific graph edge or path that the test validates. No hallucinated tests.

### API

```bash
# Recompile for deployment
curl -X POST http://localhost:8000/api/v2/compile \
  -H "Content-Type: application/json" \
  -d '{"output_dir": "./compiled_output"}'

# Generate test suite
curl -X POST http://localhost:8000/api/v2/generate-tests

# Check onboarding status
curl http://localhost:8000/api/v2/status
```

### Config (`model_config.yaml`)

```yaml
recompilation:
  backup_before_export: true
  auto_adjust_depth: true
```

---

## Pillar 4: Enterprise Security & Privacy

V2 assumes your codebase is proprietary. Every character that leaves the system boundary passes through the privacy guard.

### Content Sanitizer (`src/core/sanitizer.py`)

The `ContentSanitizer` detects and redacts sensitive data in three passes:

**Pass 1 — Secrets** (all redaction levels):
- AWS access keys (`AKIA...`) and secret keys
- GCP service account private keys
- Azure account keys
- GitHub tokens (`ghp_`, `github_pat_`)
- OpenAI keys (`sk-...`)
- Slack tokens (`xoxb-`, `xoxp-`)
- JWT tokens
- PEM private keys
- Passwords and connection strings
- Environment variable assignments

**Pass 2 — PII** (strict and moderate levels):
- Email addresses
- Phone numbers (US/international)
- Social Security Numbers
- IP addresses (unless explicitly allowed)

**Pass 3 — Infrastructure** (strict level only):
- File paths (`/home/...`, `C:\...`)
- Source code patterns (detected via AST-like heuristics)

Each redacted item is replaced with a deterministic placeholder:
```
sk-abc123... → [REDACTED:api_key:a1b2c3d4]
```

The hash suffix ensures the LLM can track that the same redacted value appears in multiple places without seeing the actual value.

### Three Redaction Levels

| Level | Secrets | PII | File Paths | Use Case |
|-------|---------|-----|------------|----------|
| `strict` | Yes | Yes | Yes | Enterprise default |
| `moderate` | Yes | Yes | No | Internal tools |
| `minimal` | Yes | No | No | Open-source projects |

### Privacy Guard (`src/core/privacy_guard.py`)

The `PrivacyGuard` wraps `LLMClient` and enforces all privacy policies:

- **Sanitization** — All text is scrubbed before any external API call
- **Content size limits** — Prevents accidental bulk data exfiltration
- **Blocked patterns** — Configurable deny-list of content that must never leave
- **Local-only mode** — Blocks ALL external API calls; uses Chroma's built-in Sentence Transformers for embeddings
- **Zero-training headers** — Sends `X-No-Store` header with every API request
- **Audit logging** — Append-only JSONL log of every external API call

### Audit Log

Every external API call is logged to `privacy_audit.jsonl`:

```json
{
  "timestamp": "2026-04-21T12:00:00Z",
  "operation": "llm_generate",
  "destination": "openai",
  "content_hash": "a1b2c3d4...",
  "content_length": 1500,
  "sanitized_length": 1420,
  "redaction_count": 3,
  "categories_found": ["api_key", "pii_email"],
  "classification": "internal",
  "blocked": false
}
```

The audit log **never** contains raw content — only a SHA-256 hash, length, and metadata about what was redacted. This makes it safe for compliance review.

### API

```bash
# View the audit log
curl http://localhost:8000/api/v2/privacy/audit

# Check privacy guard status
curl http://localhost:8000/api/v2/privacy/status
```

### Config (`model_config.yaml`)

```yaml
security:
  zero_data_retention: true

  privacy:
    local_only_mode: false        # true = block ALL external calls
    redaction_level: "strict"     # strict | moderate | minimal
    max_prompt_length: 50000
    max_embedding_length: 10000
    max_batch_size: 100
    block_source_code: true
    blocked_patterns: []          # Add internal hostnames, project names, etc.
    allow_file_paths: false
    allow_ip_addresses: false
    audit_log_path: "privacy_audit.jsonl"
    custom_redaction_patterns: [] # Enterprise-specific regex patterns
```

---

## V2 API Endpoints — Complete Reference

| Method | Endpoint | Tag | Description |
|--------|----------|-----|-------------|
| POST | `/api/v2/ingest` | onboarding | Ingest a codebase repository |
| POST | `/api/v2/ingest/yaml` | onboarding | Ingest from existing YAML directories |
| GET | `/api/v2/gaps/detect` | gap-analysis | Detect structural gaps |
| POST | `/api/v2/gaps/start-session` | gap-analysis | Start interactive gap analysis |
| GET | `/api/v2/gaps/questions` | gap-analysis | Get pending questions |
| POST | `/api/v2/gaps/answer` | gap-analysis | Submit answers, refine graph |
| POST | `/api/v2/compile` | recompilation | Recompile for deployment |
| POST | `/api/v2/generate-tests` | validation | Generate grounded test suite |
| GET | `/api/v2/status` | onboarding | Check onboarding pipeline status |
| GET | `/api/v2/privacy/audit` | security | View privacy audit log |
| GET | `/api/v2/privacy/status` | security | Check privacy guard config |

All V1 endpoints (`/api/v1/*`) continue to work unchanged.

---

## V2 Project Structure

```
src/
├── core/
│   ├── llm_client.py         # OpenAI wrapper (+ zero-training headers)
│   ├── privacy_guard.py      # Privacy enforcement gateway       ← NEW
│   └── sanitizer.py          # Content redaction engine           ← NEW
├── ingestion/                                                     ← NEW
│   ├── engine.py             # Orchestrates scan → parse → graph
│   ├── scanner.py            # File discovery and classification
│   └── parser.py             # AST-based component extraction
├── gap_analysis/                                                  ← NEW
│   ├── detector.py           # Structural gap detection
│   └── analyzer.py           # LLM-driven interactive refinement
├── recompiler/                                                    ← NEW
│   ├── recompiler.py         # Graph export and deployment prep
│   └── test_generator.py     # Grounded test case generation
├── graph/
│   ├── builder.py            # Bipartite graph (unchanged)
│   └── traverser.py          # BFS blast radius (unchanged)
├── rag/
│   ├── embedder.py           # Component → embeddings (unchanged)
│   ├── retriever.py          # Semantic search (unchanged)
│   └── vector_store.py       # Chroma wrapper (unchanged)
├── analyzer/
│   ├── impact.py             # Hybrid analysis (+ privacy guard)
│   └── reporter.py           # Risk reports (unchanged)
└── api/
    └── main.py               # FastAPI (+ V2 endpoints + privacy)
```

---

## Test Coverage

V2 adds 53 security tests on top of the existing suite:

| Test File | Count | Coverage |
|-----------|-------|----------|
| `test_security.py` | 53 | Sanitizer, privacy guard, audit logging |
| `test_graph.py` | 26 | Graph structure |
| `test_traverser.py` | 34 | BFS, cycles, chains |
| `test_impact.py` | 30 | Vector store, embedder, retriever |
| `test_analyzer.py` | varies | Scoring, reporting |
| `test_api.py` | varies | FastAPI endpoints |
| `test_ingestion.py` | varies | Scanner, parser, engine |
| `test_gap_analysis.py` | varies | Detector, analyzer sessions |
| `test_recompiler.py` | varies | Export, test generation |
| **Total** | **327** | **All pass, zero API keys needed** |
