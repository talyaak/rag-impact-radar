# Chapter 2: Embeddings Demystified — How Text Becomes Numbers

> **Reading time**: ~25 minutes
> **Prerequisites**: Chapter 1 (RAG Fundamentals)
> **After this chapter**: You'll understand what embeddings are, how they work,
> why they're the foundation of semantic search, and how to reason about them
> even though you can't visualize 1,536 dimensions.
> **Project files**: `src/rag/embedder.py`, `src/rag/vector_store.py`

---

## The Core Idea

Here's the one thing to understand about embeddings:

> **An embedding is a list of numbers that captures the MEANING of text.**

Two pieces of text with similar meaning → lists of numbers that are close together.
Two pieces of text with different meanings → lists of numbers that are far apart.

That's it. Everything else is details.

---

## Start With a Thought Experiment

Imagine you had to organize books in a library, but you could only use
a NUMBER LINE (one dimension):

```
0                                                           100
├───────────┼───────────┼───────────┼───────────┼───────────┤
Fiction                                               Non-fiction

"Harry Potter" → 12
"Lord of the Rings" → 15
"Python Cookbook" → 88
"JavaScript Guide" → 85
"Crime and Punishment" → 25
```

Notice: books with similar content get similar numbers. "Python Cookbook"
and "JavaScript Guide" are close together (88 vs 85) because they're
both programming books. "Harry Potter" and "Lord of the Rings" are close
(12 vs 15) because they're both fantasy novels.

**This number line IS a 1-dimensional embedding.** Each book is "embedded"
as a single number that captures one aspect of its meaning (fiction vs non-fiction).

### The Problem: One Dimension Isn't Enough

Where do you put a "Python for Fantasy Writers" book? It's both fiction-ish
and programming-ish. One number can't capture both aspects.

**Solution: Add more dimensions.**

```
                 Fiction ←──────→ Non-fiction
                    │
    100 ┤          │
        │  HP  LotR│         ← Fantasy fiction
     80 ┤          │
        │          │
     60 ┤          │
        │          │
     40 ┤   C&P    │            ← Literary fiction
        │          │
     20 ┤          │    PY4FW ← Python for Fantasy Writers
        │          │         (fiction + programming!)
      0 ┤──────────┼──────────
        0    20   40   60   80   100
              │          │
              │          └── Python Cookbook, JS Guide
              │              (non-fiction, programming)
              └── Fiction end

    Vertical axis: How technical (0) vs literary (100)
    Horizontal axis: How fictional (0) vs factual (100)
```

With TWO dimensions, "Python for Fantasy Writers" sits between the
fiction cluster and the programming cluster. Two numbers capture more
meaning than one.

### Real Embeddings: 1,536 Dimensions

OpenAI's `text-embedding-3-small` (which this project uses) produces
**1,536 numbers** per text. That's 1,536 different "aspects" of meaning.

You can't visualize 1,536 dimensions. Nobody can. But the math works
the same as our 2D example — texts with similar meaning get similar
lists of numbers.

```python
# What an embedding actually looks like (simplified)
embedding = [
    0.023,   # dimension 1: maybe captures "how technical is this?"
   -0.041,   # dimension 2: maybe captures "is this about security?"
    0.087,   # dimension 3: maybe captures "is this about databases?"
    ...      # 1,533 more dimensions
    0.012    # dimension 1536: some other aspect of meaning
]
```

**Important**: We don't actually know what each dimension means.
The model learned them during training. Some might correspond to
"technicality" or "security-ness," but most are abstract combinations
that humans can't name. That's fine — we don't need to interpret them,
just compare them.

---

## How Similarity Works: Cosine Similarity

Given two embeddings (lists of numbers), how do we measure "how similar"
they are?

The standard measure is **cosine similarity**. Don't let the math
intimidate you — here's the intuition:

### The Arrow Analogy

Think of each embedding as an **arrow** pointing in some direction in
high-dimensional space.

- Two arrows pointing in the **same direction** → cosine similarity = **1.0** (identical meaning)
- Two arrows pointing at **right angles** → cosine similarity = **0.0** (unrelated)
- Two arrows pointing in **opposite directions** → cosine similarity = **-1.0** (opposite meaning)

```
           Same direction (similarity ≈ 1.0)
              ↗ "password hashing"
             ↗  "cryptographic hash functions"

           Right angle (similarity ≈ 0.0)
              ↗ "password hashing"
              → "restaurant reviews"

           Opposite (similarity ≈ -1.0)
              ↗ "important"
              ↙ "unimportant"
```

### In Practice

```python
# These two texts have HIGH cosine similarity (~0.88):
text_a = "Redis-backed session persistence and invalidation"
text_b = "Managing user login sessions in a cache"

# These two texts have LOW cosine similarity (~0.15):
text_c = "Redis-backed session persistence and invalidation"
text_d = "Tax calculation for European VAT regulations"
```

The embedding model "understands" that sessions, Redis, caching, and
logins are related concepts — even though the words are different.

---

## From Theory to This Project

Let's trace exactly how embeddings flow through impact-radar.

### Step 1: What We Embed

We don't embed raw YAML. We create natural language documents first.

```
RAW YAML (bad for embedding):           NATURAL LANGUAGE (good for embedding):
┌─────────────────────────┐             ┌──────────────────────────────────────┐
│ id: auth_engine         │             │ AuthEngine (auth_engine): Core       │
│ modules:                │     →       │ identity service handling            │
│   - module_id: crypto.. │             │ authentication, authorization, and   │
│     coupling: tight     │             │ session lifecycle...                 │
│     usage: "Password.." │             │ Team: identity-team.                 │
└─────────────────────────┘             │ Criticality: critical.              │
                                        └──────────────────────────────────────┘
```

**Why?** The embedding model was trained on natural language (books, articles,
code documentation). It understands "Core identity service handling
authentication" much better than "id: auth_engine, coupling: tight".

**Project file**: `src/rag/embedder.py` — the `prepare_documents()` method
does this transformation.

### Step 2: Document Granularity

We create MULTIPLE documents per component. Here's why:

```
BAD: One giant document per component
┌──────────────────────────────────────────────────────┐
│ AuthEngine handles authentication, session lifecycle,│
│ uses crypto_utils for password hashing, uses         │
│ session_store for Redis persistence, uses event_bus  │
│ for publishing auth events, uses access_control for  │
│ RBAC permission evaluation, has endpoints for        │
│ login, logout, refresh, verify, MFA setup...         │
└──────────────────────────────────────────────────────┘
    embedding = [0.023, -0.041, 0.087, ...]
    ↑ This embedding is an AVERAGE of all those concepts.
      "Password hashing" signal is diluted by all the other text.

GOOD: Multiple focused documents per component
┌───────────────────────────────────┐
│ AuthEngine: Core identity service │  → embedding focused on "identity/auth"
│ handling authentication...        │
└───────────────────────────────────┘
┌───────────────────────────────────┐
│ AuthEngine uses crypto_utils for  │  → embedding focused on "crypto/hashing"
│ password hashing (argon2id)...    │
└───────────────────────────────────┘
┌───────────────────────────────────┐
│ AuthEngine uses session_store for │  → embedding focused on "sessions/Redis"
│ Redis-backed session persistence  │
└───────────────────────────────────┘
```

With multiple documents, a search for "password hashing" hits the
crypto_utils document with **high** similarity instead of the giant
document with **medium** similarity. More precise retrieval.

**Project file**: `src/rag/embedder.py` — creates 4 document types:
description, module_usage (one per module), api_surface, summary.
Total: 93 documents for 12 components.

### Step 3: Storing Embeddings

Once we have embeddings, we store them in Chroma (a vector database):

```
┌──────────────────────────────────────────────────────────────┐
│                    CHROMA VECTOR STORE                       │
│                                                              │
│  ID                    │ Embedding        │ Metadata          │
│  ──────────────────────┼──────────────────┼─────────────────  │
│  auth_engine__desc     │ [0.02, -0.04...] │ component: auth   │
│  auth_engine__mod__cry │ [0.31, 0.12...]  │ module: crypto    │
│  auth_engine__mod__ses │ [-0.1, 0.08...]  │ module: session   │
│  billing__desc         │ [0.15, -0.22...] │ component: bill   │
│  ...                   │ ...              │ ...               │
│  (93 documents total)  │                  │                   │
└──────────────────────────────────────────────────────────────┘
```

**Project file**: `src/rag/vector_store.py` — the `VectorStore` class
wraps Chroma with add, query, and filter methods.

### Step 4: Searching

At query time, we embed the QUERY with the same model, then find the
closest stored embeddings:

```
Query: "What components deal with user sessions?"
         │
    [same embedding model]
         │
         ▼
    [0.08, 0.15, -0.03, ...]  ← query embedding
         │
    [cosine similarity against all 93 stored embeddings]
         │
         ▼
    Results (sorted by similarity):
    1. auth_engine__module__session_store    similarity: 0.91
    2. websocket_gateway__module__session..  similarity: 0.87
    3. auth_engine__description              similarity: 0.72
    4. websocket_gateway__description        similarity: 0.68
    ...
```

**Critical rule**: The query embedding and document embeddings MUST come
from the SAME model. You can't embed documents with OpenAI and queries
with Cohere — the numbers live in different "spaces" and can't be compared.

---

## The Embedding Model: What's Inside?

You don't need to understand the internals to use embeddings, but here's
a simplified mental model:

```
┌───────────────────────────────────────────────────┐
│              EMBEDDING MODEL                       │
│     (e.g., text-embedding-3-small)                │
│                                                    │
│  Input: "Redis-backed session persistence"         │
│                                                    │
│  Internal layers (neural network):                 │
│  ┌─────────────────────────────────────────┐      │
│  │ Layer 1: Word recognition               │      │
│  │   "Redis" → cache/database concept      │      │
│  │   "session" → user state concept        │      │
│  │   "persistence" → storage concept       │      │
│  ├─────────────────────────────────────────┤      │
│  │ Layer 2: Phrase understanding            │      │
│  │   "session persistence" → keeping       │      │
│  │   user state across requests            │      │
│  ├─────────────────────────────────────────┤      │
│  │ Layer 3: Context integration            │      │
│  │   "Redis-backed session persistence" →  │      │
│  │   using Redis as a session cache for    │      │
│  │   web applications                      │      │
│  ├─────────────────────────────────────────┤      │
│  │ Layer N: Final compression              │      │
│  │   Compress all understanding into       │      │
│  │   1,536 numbers                         │      │
│  └─────────────────────────────────────────┘      │
│                                                    │
│  Output: [0.08, -0.12, 0.34, ..., 0.01]          │
│          (1,536 floats)                            │
└───────────────────────────────────────────────────┘
```

The model learned these layers by reading billions of text examples
during training. It learned that "Redis" and "Memcached" should produce
similar numbers (both are caches), while "Redis" and "tax regulations"
should produce very different numbers.

---

## Common Pitfalls (and How This Project Avoids Them)

### Pitfall 1: Embedding Raw Structured Data

```python
# BAD: Embedding raw YAML/JSON
embed("module_id: crypto_utils\ncoupling: tight\nusage: Password hashing")

# GOOD: Embedding natural language
embed("AuthEngine uses crypto_utils for password hashing (argon2id), "
      "JWT signing/verification, and TOTP secret generation for MFA. "
      "Coupling: tight.")
```

**Why**: The model understands natural language, not data formats.
"module_id: crypto_utils" is meaningless to it. "Password hashing with
argon2id" is rich with semantic signal.

### Pitfall 2: One Giant Document Per Entity

Explained above — dilutes the embedding signal. We use 4 document types
to keep each embedding focused.

### Pitfall 3: Forgetting Metadata

```python
# BAD: Embedding without metadata
store.add(text="handles authentication", id="doc1")
# Later: "Great, this matched... but what component is it from??"

# GOOD: Embedding with rich metadata
store.add(
    text="handles authentication",
    id="auth_engine__description",
    metadata={
        "component_id": "auth_engine",
        "doc_type": "description",
        "criticality": "critical",
        "team_owner": "identity-team",
    }
)
# Later: match.component_id → "auth_engine" → link to graph node
```

Metadata lets you link vector search results back to your knowledge graph
and filter queries by type.

### Pitfall 4: Mismatched Embedding Models

```python
# BAD: Different models for documents vs queries
doc_embedding = openai_embed("session management")
query_embedding = cohere_embed("user login")  # WRONG MODEL
similarity(doc_embedding, query_embedding)  # MEANINGLESS NUMBER

# GOOD: Same model for both
doc_embedding = openai_embed("session management")
query_embedding = openai_embed("user login")  # SAME MODEL
similarity(doc_embedding, query_embedding)  # MEANINGFUL: ~0.78
```

### Pitfall 5: Not Batching

```python
# BAD: One API call per document (93 calls × 100ms = 9.3 seconds)
for doc in documents:
    embedding = openai_embed(doc.text)

# GOOD: Batch API call (1 call × 100ms = 0.1 seconds)
embeddings = openai_embed_batch([doc.text for doc in documents])
```

**Project file**: `src/core/llm_client.py` — `get_embeddings_batch()` sends
up to 100 texts per API call.

---

## How Many Dimensions Do I Actually Need?

A natural question. Here's the tradeoff:

| Dimensions | Model Example | Quality | Speed | Cost |
|-----------|---------------|---------|-------|------|
| 384 | all-MiniLM-L6 | Good | Very fast | Free (local) |
| 768 | all-mpnet-base | Better | Fast | Free (local) |
| 1,536 | text-embedding-3-small | Very good | Fast | $0.02/1M tokens |
| 3,072 | text-embedding-3-large | Best | Slower | $0.13/1M tokens |

For our 93 documents, cost is irrelevant (~$0.001 total). We use 1,536
dimensions for good quality. In production with millions of documents,
you might use 384 dimensions and a local model to save cost.

---

## The Mental Model to Keep

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Text → [Embedding Model] → Numbers → [Comparison] │
│                                                     │
│  "Similar meaning" = "Numbers that are close"       │
│                                                     │
│  That's the entire concept. Everything else is      │
│  engineering: batching, storage, filtering, tuning.  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

*Previous: [01 - RAG Fundamentals](01-rag-fundamentals.md)*
*Next: [03 - Knowledge Graphs](03-knowledge-graphs.md)*
