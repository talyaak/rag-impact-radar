# Chapter 4: Retrieval Strategies — Finding the Right Needle in the Haystack

> **Reading time**: ~20 minutes
> **Prerequisites**: Chapters 1-3
> **After this chapter**: You'll understand the different ways to retrieve
> information in a RAG system, why two-stage retrieval exists, and how
> to combine structural and semantic retrieval into a hybrid pipeline.
> **Project files**: `src/rag/retriever.py`, `src/rag/vector_store.py`,
> `src/graph/traverser.py`

---

## The Retrieval Problem

You have 93 documents in your vector store and a graph with 32 nodes.
Someone asks: "What's the blast radius of changing AuthEngine?"

You need to find the RIGHT information to give to the LLM.
Too little → the LLM misses important impacts.
Too much → the LLM gets confused by irrelevant noise.

This chapter is about getting retrieval RIGHT.

---

## Strategy 1: Naive Vector Search

The simplest approach: embed the query, find the closest documents.

```
Query: "AuthEngine change impact"
         │
    [embed]
         │
         ▼
    Return top 10 closest documents

    Results:
    1. auth_engine__description          sim: 0.94  ← obviously related
    2. auth_engine__module__crypto        sim: 0.81  ← part of AuthEngine
    3. auth_engine__module__session       sim: 0.79  ← part of AuthEngine
    4. encryption_service__description    sim: 0.68  ← security-related
    5. api_gateway__module__access_ctrl   sim: 0.62  ← auth-related
    ...
```

**Problems**:
- Results 1-3 are about AuthEngine ITSELF — we already know it's affected!
- Misses WebSocketGateway (shares session_store, but description doesn't
  mention "authentication")
- Misses ReportGenerator (3 hops away — no semantic similarity at all)
- Ranks by text similarity, NOT by actual dependency strength

**Verdict**: Useful for finding semantically related things, but misses
structural dependencies entirely.

---

## Strategy 2: Pure Graph Traversal

Use BFS on the dependency graph from AuthEngine:

```
Start: AuthEngine
         │
    [BFS traversal]
         │
    Ring 1 (direct):
      EncryptionService (via crypto_utils)
      WebSocketGateway (via session_store)
      NotificationService (via event_bus)
      AuditLogger (via event_bus)
      WorkflowEngine (via event_bus + access_control)
      APIGateway (via access_control)

    Ring 2 (indirect):
      DataExporter, SearchService, BillingCore,
      PricingEngine, ReportGenerator
```

**Problems**:
- Finds ALL structurally connected components (great!)
- But doesn't capture semantic relationships that exist OUTSIDE the graph
- If two components do similar things but don't share modules, they're invisible
- No severity differentiation except by depth

**Verdict**: Catches structural blast radius perfectly, but misses
conceptual relationships.

---

## Strategy 3: Hybrid Retrieval (What This Project Does)

Combine BOTH strategies and merge results:

```
               "AuthEngine changed"
                    │         │
          ┌─────────┘         └──────────┐
          ▼                              ▼
   Graph Traversal                Semantic Search
   (structural)                   (meaning)
          │                              │
          ▼                              ▼
   ┌──────────────┐              ┌──────────────┐
   │ Depth 1:     │              │ Similarity:   │
   │ Encryption   │              │ Encryption ●  │ ← found by BOTH
   │ WebSocket    │              │ APIGateway  ●  │ ← found by BOTH
   │ Notification │              │ Audit       ●  │ ← found by BOTH
   │ AuditLogger  │              │              │
   │ Workflow     │              │              │
   │ APIGateway   │              │              │
   │ Depth 2:     │              │              │
   │ DataExporter │              │              │
   │ Search       │              │              │
   │ Billing      │              │              │
   │ Pricing      │              │              │
   │ Report       │              │              │
   └──────┬───────┘              └──────┬───────┘
          │                              │
          └──────────┬───────────────────┘
                     ▼
              MERGE & ENRICH
                     │
                     ▼
          Combined evidence set:
          - Structural paths (from graph)
          - Semantic context (from vectors)
          - Per-component severity score
```

**Why this works better than either alone**:

| What We Learn | From Graph | From Vectors |
|--------------|-----------|-------------|
| WebSocketGateway is affected | via session_store (structural) | probably not (different words) |
| EncryptionService is affected | via crypto_utils (structural) | via "security" similarity (semantic) |
| The REASON for impact | shared module + usage description | similar domain concepts |
| HOW STRONGLY impacted | coupling metadata (tight/loose) | similarity score (0-1) |

---

## Two-Stage Retrieval: Cast Wide, Then Refine

Our retriever uses a two-stage approach. This is standard in production
RAG systems:

```
            ┌──────────────────────────────┐
            │      STAGE 1: Recall         │
            │                              │
            │  Goal: Don't miss anything   │
            │  Method: top_k = 10          │
            │  Trade-off: May include      │
            │  some irrelevant results     │
            │                              │
            │  "Cast a wide net"           │
            └──────────────┬───────────────┘
                           │
                  10 candidate results
                           │
                           ▼
            ┌──────────────────────────────┐
            │     STAGE 2: Precision       │
            │                              │
            │  Goal: Keep only the best    │
            │  Method: rerank_top_k = 5    │
            │  Trade-off: Might lose       │
            │  some borderline results     │
            │                              │
            │  "Keep the good catch"       │
            └──────────────┬───────────────┘
                           │
                   5 high-quality results
                           │
                           ▼
                   Final retrieval set
```

**Why two stages?**

Think of it like hiring:
- **Stage 1** (resume screening): Review 100 resumes, keep 20 that look promising.
  Optimize for NOT MISSING good candidates (recall).
- **Stage 2** (interview): Interview the 20, hire 5.
  Optimize for SELECTING the best (precision).

If you only did Stage 2 with 100 candidates, it'd take forever.
If you only did Stage 1, you'd hire based on resumes alone (unreliable).

In RAG terms:
- **Stage 1**: Vector similarity search returns top_k=10 candidates. Fast but rough.
- **Stage 2**: Reranker scores each candidate more carefully, keeps top 5. Slower but precise.

**Config** (`config/model_config.yaml`):
```yaml
retrieval:
  top_k: 10               # Stage 1: retrieve 10 candidates
  similarity_threshold: 0.7  # Minimum similarity to include
  rerank: true             # Enable Stage 2
  rerank_top_k: 5          # Stage 2: keep best 5
```

---

## Metadata Filtering: Scoping Your Search

Vector search can be scoped using metadata filters. This is like adding
a WHERE clause to a similarity search:

```sql
-- Traditional database:
SELECT * FROM components WHERE team = 'security-team' ORDER BY name;

-- Vector search (no filter):
"Find documents similar to 'password hashing'"
→ Returns: auth_engine__crypto, encryption_service__crypto, billing__cache...

-- Vector search (with metadata filter):
"Find documents similar to 'password hashing' WHERE doc_type = 'module_usage'"
→ Returns: ONLY module_usage documents about password hashing
   (skips descriptions, summaries, API surfaces)
```

**Common filter patterns in this project**:

```python
# Find module-level matches only
results = store.query(
    query_text="session management",
    where={"doc_type": "module_usage"}
)

# Exclude a specific component
results = store.query(
    query_text="authentication",
    where={"component_id": {"$ne": "auth_engine"}}
)
```

**Project file**: `src/rag/vector_store.py` — the `query()` method accepts
`where` filters that map directly to Chroma's filtering syntax.

---

## From Documents to Components: Aggregation

Vector search returns DOCUMENTS, but we think in COMPONENTS.
AuthEngine might have 8 documents in the store. If 3 of them match
our query, we need to aggregate:

```
Raw vector results:                 Aggregated by component:
┌─────────────────────────────┐     ┌──────────────────────────────┐
│ auth_engine__desc    0.88   │     │ AuthEngine:                  │
│ auth_engine__crypto  0.82   │ →   │   best_score: 0.88           │
│ auth_engine__session 0.71   │     │   matching_docs: 3           │
│ encrypt__desc        0.79   │     │   avg_score: 0.80            │
│ encrypt__crypto      0.74   │     │                              │
│ gateway__access      0.65   │     │ EncryptionService:           │
└─────────────────────────────┘     │   best_score: 0.79           │
                                    │   matching_docs: 2           │
                                    │                              │
                                    │ APIGateway:                  │
                                    │   best_score: 0.65           │
                                    │   matching_docs: 1           │
                                    └──────────────────────────────┘
```

**Why aggregate?**
- The impact analyzer needs per-component decisions, not per-document
- Multiple matching documents from one component = stronger signal
- Deduplication prevents one component from dominating results

**Project file**: `src/rag/retriever.py` — `_aggregate_by_component()`

---

## The Similarity Threshold: Where to Draw the Line

```
Similarity:  1.0                    0.7            0.0
             │                       │              │
             ▼                       ▼              ▼
    "Exact match"            "Probably related"   "Unrelated"
        ●●●●                    ●●                  ●●●●●●●
     (keep these)           (threshold)          (discard these)
                                 ↑
                          similarity_threshold = 0.7
```

**Too high** (0.9): Only finds near-exact text matches. Misses
paraphrased or conceptually related documents.

**Too low** (0.3): Returns everything vaguely related. The LLM
gets confused by irrelevant context.

**Our default** (0.7): Balances precision and recall. In practice,
you'd evaluate this empirically with a test set of queries.

**How to tune it**: Create 20 test queries where you KNOW what the
correct results should be. Run retrieval at different thresholds.
Measure precision (% of results that are correct) and recall
(% of correct results that were found). Pick the threshold that
maximizes both.

---

## Putting It All Together: A Complete Retrieval Flow

When someone asks "What breaks if I change AuthEngine and BillingCore?":

```
Step 1: GRAPH TRAVERSAL
  Start: [AuthEngine, BillingCore]
  BFS with max_depth=5, include_weak_coupling=true
  Result:
    Direct (depth 1): EncryptionService, WebSocketGateway,
      NotificationService, AuditLogger, WorkflowEngine,
      APIGateway, PricingEngine, SearchService, ReportGenerator
    Indirect (depth 2+): DataExporter
    Paths: 11 impact paths recorded
    Affected variants: all 8

Step 2: SEMANTIC SEARCH (for each changed component)
  Query: AuthEngine's description → find similar components
  Query: BillingCore's description → find similar components
  Exclude: AuthEngine, BillingCore (already known)
  Result:
    EncryptionService (high similarity — both security-related)
    APIGateway (moderate similarity — both handle access control)

Step 3: MERGE
  Union of graph results + semantic results
  Deduplicate by component_id
  For each component:
    - Structural evidence: impact paths from graph
    - Semantic evidence: similarity scores from vectors
    - Coupling severity: from edge metadata

Step 4: AUGMENT PROMPT
  "Given these impacts:
   [Direct] EncryptionService via crypto_utils (tight, password hashing)
   [Direct] WebSocketGateway via session_store (tight, session tracking)
   [Indirect] DataExporter via EncryptionService→key_rotation (tight)
   ...
   Explain the risk to each affected product variant."

Step 5: GENERATE
  LLM produces human-readable risk report grounded in the evidence.
```

---

## Retrieval Anti-Patterns

Things that make retrieval worse:

### 1. Retrieving Too Much
```
# BAD: top_k=100 for a 12-component system
# The LLM gets 100 documents, most irrelevant

# GOOD: top_k=10, rerank to 5
# Focused, high-quality context
```

### 2. No Exclusion of Known Items
```
# BAD: Query "AuthEngine" → top result is AuthEngine itself
# Wastes a retrieval slot on something we already know

# GOOD: Exclude changed components from results
retriever.find_related_components(
    query="authentication",
    exclude_component_ids=["auth_engine"]  # Already know this one
)
```

### 3. Flat Retrieval (No Aggregation)
```
# BAD: Return 10 documents, 6 are about the same component
# One component dominates, others are drowned out

# GOOD: Aggregate by component, then rank components
# Each component gets fair representation
```

### 4. Ignoring Retrieval Quality
```
# BAD: "The LLM's output is wrong, let's use a better model"
# (The real problem: retrieval returned wrong context)

# GOOD: First check if retrieval returned the right documents
# Fix retrieval BEFORE tuning generation
```

---

## Key Takeaways

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  1. Hybrid retrieval (graph + vectors) > either alone        │
│     Structural + semantic = comprehensive coverage           │
│                                                              │
│  2. Two-stage retrieval: recall first, precision second      │
│     Cast wide (top_k=10), then refine (rerank to 5)         │
│                                                              │
│  3. Metadata filters scope your search                       │
│     WHERE doc_type = "module_usage" avoids noise             │
│                                                              │
│  4. Aggregate documents → entities                           │
│     The LLM needs component-level decisions                  │
│                                                              │
│  5. Retrieval quality bounds generation quality               │
│     Fix retrieval first, generation second                    │
│                                                              │
│  6. The similarity threshold is the key tuning knob          │
│     Too high = misses things. Too low = noise.               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

*Previous: [03 - Knowledge Graphs](03-knowledge-graphs.md)*
*Next: [05 - Grounded Generation](05-grounded-generation.md)*
