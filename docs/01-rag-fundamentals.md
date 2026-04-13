# Chapter 1: RAG Fundamentals — What It Is and Why It Exists

> **Reading time**: ~20 minutes
> **Prerequisites**: You use AI chatbots (ChatGPT, Claude) daily. That's it.
> **After this chapter**: You'll understand what RAG is, why every serious AI
> application uses it, and how it solves the biggest problem in AI.

---

## The Problem RAG Solves

You've used ChatGPT or Claude. You've noticed something:

**They make stuff up.**

Ask Claude about your company's internal billing system and it'll confidently
describe one — except it'll be completely fictional. It sounds right. It uses
the right jargon. But it's fabricated from patterns in its training data.

This is called **hallucination**, and it's the #1 problem in applied AI.

Here's why it happens:

```
┌─────────────────────────────────────────────────────┐
│                    THE LLM'S BRAIN                  │
│                                                     │
│  Training data: books, websites, code (up to 2024)  │
│  ┌───────────────────────────────────────────────┐  │
│  │ Knows: Python syntax, general CS concepts,    │  │
│  │        how billing systems typically work      │  │
│  │                                               │  │
│  │ Does NOT know: YOUR billing system, YOUR      │  │
│  │        components, YOUR dependency chains     │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  When asked about YOUR system → fills in gaps       │
│  with plausible-sounding fabrications               │
└─────────────────────────────────────────────────────┘
```

**RAG fixes this by giving the LLM a cheat sheet before it answers.**

---

## RAG in One Sentence

> **RAG = Look up the facts first, then ask the AI to explain them.**

That's it. The entire field in one sentence. Everything else is implementation
details.

---

## RAG in One Diagram

```
                    YOUR QUESTION
                         │
                         ▼
              ┌─────────────────────┐
              │   1. RETRIEVAL      │   "Find relevant facts"
              │                     │
              │  Search your data:  │
              │  • Vector database  │
              │  • Knowledge graph  │
              │  • Traditional DB   │
              └─────────┬───────────┘
                        │
                  Retrieved facts
                        │
                        ▼
              ┌─────────────────────┐
              │   2. AUGMENTATION   │   "Add facts to the prompt"
              │                     │
              │  "Given these facts │
              │   about our system: │
              │   [RETRIEVED DATA]  │
              │                     │
              │   Explain: ..."     │
              └─────────┬───────────┘
                        │
                  Augmented prompt
                        │
                        ▼
              ┌─────────────────────┐
              │   3. GENERATION     │   "AI explains the facts"
              │                     │
              │  LLM writes answer  │
              │  grounded in the    │
              │  retrieved facts    │
              └─────────┬───────────┘
                        │
                        ▼
                 GROUNDED ANSWER
             (facts + explanation,
              not hallucination)
```

**R**etrieval → **A**ugmentation → **G**eneration = **RAG**

---

## A Real Analogy: The Open-Book Exam

Think of it like school exams:

| Exam Type | AI Equivalent | Quality |
|-----------|---------------|---------|
| **Closed-book** | Raw LLM (no RAG) | Student writes from memory. Gets the gist right but invents specific details. |
| **Open-book** | LLM with RAG | Student looks up facts in the textbook, then writes an answer using those facts. Much more accurate. |

The LLM is a brilliant student who read every textbook ever written — but your
internal systems weren't in any textbook. RAG gives it the textbook for YOUR system.

---

## Why Not Just Put Everything in the Prompt?

You might think: "Why not just paste all my data into the AI prompt?"

Three reasons:

### 1. Context Windows Have Limits

Even the largest LLMs have a finite "working memory" (context window):

```
GPT-4o:       128,000 tokens  ≈  ~96,000 words  ≈  ~200 pages
Claude Opus:  200,000 tokens  ≈ ~150,000 words   ≈  ~300 pages

Your codebase:  Could be millions of lines
Your docs:      Could be thousands of pages
Your data:      Could be terabytes
```

You physically cannot fit everything into one prompt.

### 2. More Context = Worse Performance

Even if you COULD fit everything, you shouldn't. Research shows that
LLMs perform worse when given irrelevant information — they get
distracted, just like humans. This is called the "lost in the middle"
problem.

```
Performance vs. Context Size:

Quality │ ●●●
        │    ●●●
        │       ●●●
        │          ●●               Sweet spot: just the
        │            ●●             relevant facts (RAG)
        │              ●●
        │                ●●●●●●    Everything dumped in
        │                          (diminishing returns)
        └──────────────────────── Context size
```

### 3. Retrieval Is Reusable

Once you build a RAG system, it works for ANY question about your data.
Manually copying relevant context into each prompt doesn't scale.

---

## The Two Types of Retrieval

RAG systems typically use two retrieval methods. Think of them as
two different ways to search a library:

### Type 1: Semantic Search (Vector-Based)

**Analogy**: A librarian who understands MEANING.

You ask: "I need something about keeping users logged in"
Librarian finds: A book titled "Session Persistence and Token Management"

The words don't match, but the MEANING matches. This is what vector
search does — it converts text into numbers (embeddings) that capture
meaning, then finds the closest matches.

```
Your query:     "keeping users logged in"
                         │
                    [embedding]
                         │
                         ▼
                   [0.23, -0.41, 0.87, ...]   ← numbers that mean
                         │                        "user session stuff"
                         │
            Compare with all stored embeddings
                         │
                         ▼
    ┌─────────────────────────────────────────────┐
    │ Match 1: "session_store: Redis-backed       │  similarity: 0.92
    │          session persistence"                │
    │ Match 2: "AuthEngine: session lifecycle"     │  similarity: 0.88
    │ Match 3: "WebSocketGateway: connection       │  similarity: 0.71
    │          tracking"                           │
    └─────────────────────────────────────────────┘
```

**In this project**: `src/rag/vector_store.py` and `src/rag/retriever.py`
implement semantic search using Chroma and OpenAI embeddings.

### Type 2: Structural Search (Graph-Based)

**Analogy**: An engineer who understands ARCHITECTURE.

You ask: "What breaks if I change AuthEngine?"
Engineer follows the dependency map: "AuthEngine shares `session_store` with
WebSocketGateway, shares `event_bus` with AuditLogger..."

This isn't about meaning — it's about structure. The engineer follows
concrete connections in a dependency graph.

```
Your change:    AuthEngine modified
                      │
                 [graph traversal]
                      │
          ┌───────────┼──────────────┐
          │           │              │
          ▼           ▼              ▼
    crypto_utils  session_store  event_bus
          │           │              │
          ▼           ▼              ├──────────┐
    Encryption   WebSocket      AuditLogger  Workflow
    Service      Gateway                     Engine
```

**In this project**: `src/graph/builder.py` and `src/graph/traverser.py`
implement structural search using NetworkX.

### Why We Use Both

Neither method alone is complete:

| Method | Finds | Misses |
|--------|-------|--------|
| **Semantic search** | Components that SOUND related ("encryption" ↔ "security") | Structural dependencies invisible in text |
| **Graph traversal** | Components that ARE connected (via shared modules) | Conceptual similarities not captured by edges |

**Hybrid retrieval** = Use both, merge results, give everything to the LLM.

```
          Semantic search results     Graph traversal results
                    │                          │
                    └──────────┬───────────────┘
                               │
                          MERGE & DEDUPLICATE
                               │
                               ▼
                    Combined evidence set
                               │
                               ▼
                    LLM generates explanation
                    grounded in ALL evidence
```

**In this project**: The impact analyzer (Phase 5+6, coming next) merges
graph traversal results with semantic retrieval results.

---

## The RAG Pipeline in This Project

Here's exactly how impact-radar implements RAG:

```
        ┌──────────────────────────────────────────────────┐
        │                 OFFLINE (once)                    │
        │                                                  │
        │  YAML files ──→ Graph Builder ──→ NetworkX graph │
        │       │                                          │
        │       └──→ Embedder ──→ Chroma vector store      │
        └──────────────────────────────────────────────────┘

        ┌──────────────────────────────────────────────────┐
        │              ONLINE (per query)                   │
        │                                                  │
        │  "What breaks if AuthEngine changes?"            │
        │          │                  │                     │
        │          ▼                  ▼                     │
        │   Graph Traversal    Semantic Search              │
        │     (structural)       (meaning)                  │
        │          │                  │                     │
        │          └──────┬───────────┘                     │
        │                 ▼                                 │
        │          Merged evidence                          │
        │                 │                                 │
        │                 ▼                                 │
        │        LLM Risk Explanation                      │
        │                 │                                 │
        │                 ▼                                 │
        │         Risk Report per Variant                   │
        └──────────────────────────────────────────────────┘
```

---

## The 5 Laws of RAG

As you build RAG systems, these principles always apply:

### Law 1: Garbage In, Garbage Out
If your stored documents are poorly written, retrieval returns bad context,
and the LLM generates bad answers. Document preparation is 60% of RAG quality.
(See: `src/rag/embedder.py` — we carefully craft natural language documents
from structured YAML, not just dump raw data.)

### Law 2: Retrieval Bounds Generation
The LLM can only explain what was retrieved. If retrieval misses a fact,
no amount of LLM sophistication will recover it. Optimize retrieval first,
generation second.

### Law 3: More Relevant Context > More Context
5 highly relevant paragraphs beat 50 vaguely related pages. This is why
we use similarity thresholds and reranking.

### Law 4: Structure + Semantics > Either Alone
Combine knowledge graphs with vector search. Neither is sufficient.
Graph catches structural relationships. Vectors catch semantic similarity.

### Law 5: Ground Everything
Never ask the LLM to infer facts. Give it facts and ask it to explain them.
"AuthEngine impacts WebSocketGateway via session_store" → explain why.
NOT: "Does AuthEngine impact WebSocketGateway?" → the LLM might say no.

---

## Vocabulary Cheat Sheet

Keep this handy as you read the rest of the docs:

| Term | Plain English | In This Project |
|------|--------------|-----------------|
| **Embedding** | A list of numbers that captures the meaning of text | OpenAI's `text-embedding-3-small` converts descriptions to 1536 numbers |
| **Vector store** | A database optimized for "find similar things" searches | Chroma stores and searches our component embeddings |
| **Cosine similarity** | A 0-1 score of how similar two embeddings are | 1.0 = identical meaning, 0.0 = unrelated |
| **Top-k** | "Return the k most similar results" | We retrieve top 10, then rerank to top 5 |
| **Chunk** | A piece of a document small enough to embed meaningfully | Each module_usage entry is one chunk |
| **Collection** | A group of embeddings in a vector store (like a DB table) | `impact_radar_components` in Chroma |
| **Grounding** | Giving the LLM facts so it explains rather than invents | Our impact paths ARE the grounding data |
| **Hallucination** | When the LLM confidently states false information | What we prevent by using RAG |
| **Bipartite graph** | A graph with two types of nodes, edges only between types | Components ↔ Modules in our dependency graph |
| **BFS** | Breadth-First Search — explore closest nodes first | How we traverse the dependency graph |
| **Hybrid retrieval** | Combining vector search + graph traversal | The core innovation of this project |

---

## What's Next

Now that you understand WHAT RAG is and WHY it exists, the next chapters
dive deep into each component:

- **Chapter 2**: Embeddings — How text becomes numbers (and why that's magic)
- **Chapter 3**: Knowledge Graphs — How structure captures what meaning can't
- **Chapter 4**: Retrieval Strategies — Finding the right needle in the haystack
- **Chapter 5**: Grounded Generation — Making the LLM explain facts, not invent them
- **Chapter 6**: Architecture Walkthrough — How every file in this project connects

---

*Next chapter: [02 - Embeddings Demystified](02-embeddings-demystified.md)*
