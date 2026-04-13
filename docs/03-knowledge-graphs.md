# Chapter 3: Knowledge Graphs — How Structure Captures What Meaning Can't

> **Reading time**: ~25 minutes
> **Prerequisites**: Chapter 1 (RAG Fundamentals)
> **After this chapter**: You'll understand why vector search isn't enough,
> how dependency graphs work, what "bipartite" means, and why BFS is the
> right algorithm for impact analysis.
> **Project files**: `src/graph/builder.py`, `src/graph/traverser.py`

---

## Why Vectors Aren't Enough

Chapter 2 showed how embeddings capture meaning. That's powerful.
But here's what embeddings CAN'T do:

```
Question: "If I change AuthEngine, does WebSocketGateway break?"

Vector search says:
  - AuthEngine description: talks about "authentication, sessions"
  - WebSocketGateway description: talks about "WebSocket, real-time, connections"
  - Similarity: ~0.35 (low — different words, different concepts)
  - Answer: "Probably not related" ← WRONG!

Graph traversal says:
  - AuthEngine uses session_store
  - WebSocketGateway ALSO uses session_store
  - They share a dependency: session_store
  - A session_store schema change breaks BOTH
  - Answer: "Direct dependency via shared module" ← CORRECT!
```

Embeddings capture **what things mean**. Graphs capture **how things connect**.
You need both.

---

## Graphs 101: Nodes and Edges

A graph is simple: **dots** (nodes) connected by **lines** (edges).

```
    AuthEngine ──── crypto_utils ──── EncryptionService
         │
    session_store
         │
    WebSocketGateway
```

- **Nodes**: AuthEngine, crypto_utils, session_store, WebSocketGateway, EncryptionService
- **Edges**: The lines connecting them (dependencies)

That's a graph. No PhD required.

### Directed vs. Undirected

```
Directed (one-way arrows):       Undirected (two-way connections):
  A ──→ B                          A ──── B
  A depends on B,                  A and B are connected.
  but B doesn't depend on A.       The relationship is mutual.
```

**This project uses UNDIRECTED edges** because if AuthEngine and
WebSocketGateway both use session_store, the impact flows both ways.
A change to session_store affects both. A change to AuthEngine that
modifies how it uses session_store could affect WebSocketGateway too.

---

## What's a Bipartite Graph?

Our graph has a special structure called **bipartite**. It means:

> **Two types of nodes, and edges ONLY connect different types.**

```
COMPONENTS (type 1):         MODULES (type 2):

  AuthEngine ─────────────── crypto_utils
       │ ╲                        │
       │  ╲                       │
       │   ╲                 EncryptionService
       │    ╲
  session_store ──────────── WebSocketGateway
       │
  event_bus ─────────────── AuditLogger
       │                        │
       │                   query_builder
       │                        │
  NotificationService      SearchService
```

Rules of our bipartite graph:
- Components connect to modules (AuthEngine → crypto_utils)
- Modules connect to components (crypto_utils → EncryptionService)
- Components NEVER connect directly to components
- Modules NEVER connect directly to modules

### Why Bipartite?

Because it mirrors reality. In a software system:

- AuthEngine doesn't depend on EncryptionService DIRECTLY
- AuthEngine depends on `crypto_utils` (a shared library/module)
- EncryptionService ALSO depends on `crypto_utils`
- The shared module IS the reason they're coupled

A bipartite graph preserves this "WHY" — the module node in the middle
is the EXPLANATION for why two components are related. This is gold for
the LLM later: instead of saying "AuthEngine impacts EncryptionService,"
we say "AuthEngine impacts EncryptionService because they share
crypto_utils, which AuthEngine uses for password hashing and
EncryptionService uses for AES-256-GCM encryption."

```
Without module nodes (less useful):
  AuthEngine ───── EncryptionService
  (we know they're related, but WHY?)

With module nodes (more useful):
  AuthEngine ── crypto_utils ── EncryptionService
  (they share crypto_utils — that's the explanation)
```

---

## Edge Metadata: Not All Dependencies Are Equal

Each edge in our graph carries extra information:

```
AuthEngine ──[tight, "Password hashing"]──→ crypto_utils
AuthEngine ──[loose, "Publishes auth events"]──→ event_bus
```

| Field | Meaning | Values |
|-------|---------|--------|
| **coupling** | How tightly bound the dependency is | `tight`, `loose`, `optional` |
| **usage** | Human-readable description of WHY | "Password hashing (argon2id)..." |

**Why does coupling matter?**

```
TIGHT coupling:
  AuthEngine → crypto_utils
  "If crypto_utils changes its hashing API, AuthEngine BREAKS."
  Risk: HIGH. Must test immediately.

LOOSE coupling:
  AuthEngine → event_bus
  "If event_bus changes its message format, AuthEngine's core auth
   still works, but audit logging might miss events."
  Risk: MEDIUM. Should test, but not a showstopper.

OPTIONAL coupling:
  SomeComponent → metrics_collector
  "If metrics_collector is down, the component still works fine.
   You just lose observability."
  Risk: LOW. Can defer testing.
```

The traverser uses coupling to compute impact severity. A path through
all-tight couplings is more dangerous than a path through loose ones.

**Project file**: See `src/graph/builder.py` — edges store `coupling`
and `usage` as attributes:
```python
self._graph.add_edge(
    comp.id, module_id,
    coupling=coupling,   # "tight", "loose", or "optional"
    usage=usage,         # "Password hashing (argon2id)..."
)
```

---

## Graph Traversal: How We Find the Blast Radius

Now the key question: **given that AuthEngine changed, what else is affected?**

This is a graph traversal problem. We start at AuthEngine and walk
outward through the graph, finding every reachable component.

### The Algorithm: Breadth-First Search (BFS)

BFS explores the graph in **rings** expanding outward from the start:

```
                         START: AuthEngine
                               │
            ┌──────────────────┼──────────────────────┐
   Ring 1   │                  │                      │
  (depth 1) │                  │                      │
            ▼                  ▼                      ▼
       crypto_utils      session_store           event_bus
            │                  │               ┌────┼────┐
            ▼                  ▼               ▼    ▼    ▼
      Encryption         WebSocket         Notif. Audit  Workflow
      Service            Gateway           Svc    Logger Engine
            │                                      │
   Ring 2   │                                      │
  (depth 2) ▼                                      ▼
       key_rotation                           query_builder
            │                                      │
            ▼                                      ▼
       DataExporter                          SearchService

   Ring 3                                          │
  (depth 3)                                        ▼
                                              db_connector
                                                   │
                                                   ▼
                                             ReportGenerator
```

### Why BFS and Not DFS?

**BFS (Breadth-First Search)**: Explore all depth-1 neighbors first,
then all depth-2, then depth-3...

**DFS (Depth-First Search)**: Go as deep as possible down one path,
then backtrack and try another.

```
BFS order: Ring 1, Ring 1, Ring 1, Ring 2, Ring 2, Ring 3...
DFS order: Path 1 (deep), backtrack, Path 2 (deep), backtrack...
```

BFS is correct for impact analysis because:

1. **Severity ordering**: Direct dependencies (depth 1) are higher risk
   than indirect ones (depth 3). BFS discovers them in risk order.

2. **Depth classification**: BFS naturally tells us "EncryptionService is
   depth 1 (direct)" vs "DataExporter is depth 2 (indirect)". DFS would
   discover them in arbitrary order.

3. **Depth limiting**: "Only show me impacts within 3 hops" is trivial
   with BFS — just stop expanding at depth 3.

4. **Shortest path guarantee**: If a component is reachable via multiple
   paths, BFS finds the shortest one first. This gives us the most
   relevant impact path for the LLM to explain.

### The Cycle Problem

Look at this chain from our seed data:

```
AuthEngine → event_bus → WorkflowEngine → access_control → AuthEngine
   ↑                                                           │
   └───────────────────────────────────────────────────────────┘
                        IT'S A CYCLE!
```

If we follow every edge naively, we'd go around this loop forever.

**Solution**: The **visited set**. Once we've reached a component, we
never process it again:

```python
visited_components = set(["auth_engine"])  # Start with changed components

# When we reach workflow_engine via event_bus:
visited_components.add("workflow_engine")  # Mark as visited

# Later, when following access_control back to auth_engine:
if "auth_engine" in visited_components:
    continue  # SKIP — already visited, prevents infinite loop
```

**Project file**: `src/graph/traverser.py` — the BFS loop:
```python
while queue:
    current_comp, depth, path = queue.popleft()
    for module_id in graph.neighbors(current_comp):
        for neighbor_comp in graph.neighbors(module_id):
            if neighbor_comp in visited_components:
                continue  # Cycle detected — skip
            visited_components.add(neighbor_comp)
            queue.append((neighbor_comp, depth + 1, new_path))
```

---

## Impact Paths: The Evidence Trail

The traverser doesn't just find WHAT is affected — it records HOW
the impact propagates. This is the **impact path**:

```
Direct impact (depth 1):
  AuthEngine → crypto_utils → EncryptionService
  [3 nodes in the path]

Indirect impact (depth 2):
  AuthEngine → crypto_utils → EncryptionService → key_rotation → DataExporter
  [5 nodes in the path]

Indirect impact (depth 3):
  AuthEngine → event_bus → AuditLogger → query_builder → SearchService
  [5 nodes in the path]
```

**Why record paths?** Because the LLM needs EVIDENCE.

Without paths:
> "AuthEngine impacts DataExporter."
> (LLM: "Hmm, how? I'll guess... maybe because of data export authentication?")
> ← HALLUCINATION

With paths:
> "AuthEngine impacts DataExporter via: AuthEngine uses crypto_utils for
> password hashing → EncryptionService uses crypto_utils for AES-256
> encryption → EncryptionService uses key_rotation for automated key
> lifecycle → DataExporter uses key_rotation for encrypting export files."
> ← GROUNDED EXPLANATION

---

## From Components to Variants: The Business Impact

Finding affected components isn't the end goal. Product managers and
customers think in terms of **product variants** (plans/tiers), not
internal components.

```
Graph traversal output:       Business impact:
  AuthEngine changed          "Which customer plans are affected?"
  → EncryptionService         
  → WebSocketGateway          Starter Global:     5 components, 2 affected
  → NotificationService       Professional EU:   10 components, 4 affected
  → AuditLogger               Enterprise Global: 12 components, 7 affected
  → WorkflowEngine            Platform:           6 components, 3 affected
  → APIGateway
```

The traverser maps affected components → affected variants:

```python
for variant in all_variants:
    affected_in_variant = variant.components & all_affected_components
    if affected_in_variant:
        report(variant, affected_in_variant)
```

**Project file**: `src/graph/traverser.py` — `_compute_affected_variants()`

---

## The Four Dependency Chains in Our Seed Data

Our seed data was designed with 4 non-obvious dependency chains.
Here's each one explained as a story:

### Chain 1: The Internationalization Surprise

```
NotificationService → i18n_utils ← SearchService → cache_layer ← APIGateway
```

**Story**: The i18n team ships a fix to locale fallback logic in `i18n_utils`.
"Just a translation fix," they say. But SearchService also uses i18n_utils
for multilingual query tokenization. If the fallback logic change causes
SearchService to generate different cache keys, it could poison the shared
`cache_layer` — which APIGateway also reads from. A "translations fix"
cascades to API availability.

### Chain 2: The Billing-to-Security Pipeline

```
BillingCore → currency_utils ← PricingEngine → rate_limiter ← APIGateway
→ config_loader ← EncryptionService → key_rotation ← DataExporter
```

**Story**: Finance adds support for a new currency in `currency_utils`.
PricingEngine also uses currency_utils and starts returning prices in
the new format. PricingEngine shares `rate_limiter` with APIGateway,
and if PricingEngine's response format changes, APIGateway's caching
might behave differently. APIGateway shares `config_loader` with
EncryptionService, which shares `key_rotation` with DataExporter.
A currency change potentially requires testing across billing, pricing,
API routing, encryption, AND data export.

### Chain 3: The Audit Trail Cycle (circular!)

```
AuthEngine → event_bus → AuditLogger → query_builder → SearchService
→ db_connector → ReportGenerator → job_scheduler → WorkflowEngine
→ access_control → AuthEngine (BACK TO START)
```

**Story**: This is a full circle. ANY component in this ring can trigger
a full-ring assessment. Change the event_bus message format and it ripples
through AuditLogger, into SearchService, into ReportGenerator, into
WorkflowEngine, and back into AuthEngine. The cycle means there's no
"safe" component — everything is entangled.

### Chain 4: The Real-Time Communication Trap (circular!)

```
WebSocketGateway → session_store → AuthEngine → crypto_utils
→ EncryptionService → config_loader → BillingCore → metrics_collector
→ WebSocketGateway (BACK TO START)
```

**Story**: A WebSocket protocol change seems contained to real-time
features. But WebSocketGateway shares session_store with AuthEngine.
AuthEngine shares crypto_utils with EncryptionService. EncryptionService
shares config_loader with BillingCore. BillingCore shares
metrics_collector back with WebSocketGateway. Full circle spanning
real-time, auth, encryption, and billing.

---

## Knowledge Graphs in the Wider RAG World

Our bipartite component-module graph is one type of knowledge graph.
Here are others you'll encounter:

| Domain | Nodes | Edges | Use Case |
|--------|-------|-------|----------|
| **This project** | Components, Modules | "uses" | Impact analysis |
| **Wikipedia** | Articles | "links to" | Question answering |
| **Code intelligence** | Functions, Files, Classes | "calls", "imports" | Code search |
| **Medical** | Diseases, Symptoms, Drugs | "causes", "treats" | Diagnosis |
| **Supply chain** | Suppliers, Parts, Products | "provides", "contains" | Risk assessment |

The pattern is always the same:
1. Build a graph from structured data
2. Traverse it to find structural relationships
3. Combine with semantic search for completeness
4. Let the LLM explain the findings

This pattern is called **GraphRAG** and it's one of the most active
research areas in AI (2024-2025).

---

## Key Takeaways

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  1. Graphs capture STRUCTURE, embeddings capture MEANING   │
│     Neither alone is complete.                             │
│                                                            │
│  2. Bipartite structure preserves the WHY                  │
│     (the shared module IS the explanation)                 │
│                                                            │
│  3. BFS finds impacts in severity order                    │
│     (close impacts first, distant impacts later)           │
│                                                            │
│  4. The visited set prevents infinite loops on cycles      │
│     (essential — real dependency graphs have cycles)        │
│                                                            │
│  5. Impact paths are LLM evidence                          │
│     (the graph provides facts, the LLM explains them)      │
│                                                            │
│  6. Edge metadata (coupling, usage) enables severity       │
│     scoring and grounded generation                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

*Previous: [02 - Embeddings Demystified](02-embeddings-demystified.md)*
*Next: [04 - Retrieval Strategies](04-retrieval-strategies.md)*
