# RAG Impact Radar — Zero to Hero Curriculum

This six-chapter guide teaches Retrieval-Augmented Generation (RAG) from
scratch by building a real tool: **Impact Radar**, a Change Impact Analyzer
that combines vector search with knowledge graphs to answer questions like
"What breaks if I change AuthEngine?" You do not need prior ML or NLP
experience — just daily familiarity with AI assistants like ChatGPT or Claude.

---

## Who This Is For

You use AI every day, you have seen it hallucinate, and you want to understand
the engineering pattern that fixes that. Maybe you are a developer evaluating
RAG for a product, a technical PM scoping a project, or simply curious about
how grounded AI systems work. Each chapter is self-contained reading with
diagrams and code pointers — no setup required to learn the concepts.

---

## Curriculum Map

| # | Chapter | Time | Key Concepts | Project Files |
|---|---------|------|--------------|---------------|
| 1 | [RAG Fundamentals](01-rag-fundamentals.md) | ~20 min | Hallucination problem, open-book exam analogy, the RAG pipeline, 5 Laws of RAG | — (conceptual overview) |
| 2 | [Embeddings Demystified](02-embeddings-demystified.md) | ~25 min | Text-to-numbers, 1D to 1536D, cosine similarity, embedding pitfalls | `src/rag/embedder.py`, `src/rag/vector_store.py` |
| 3 | [Knowledge Graphs](03-knowledge-graphs.md) | ~25 min | Bipartite graphs, edge metadata, BFS traversal, cycle detection, dependency chains | `src/graph/builder.py`, `src/graph/traverser.py` |
| 4 | [Retrieval Strategies](04-retrieval-strategies.md) | ~20 min | Naive vs graph vs hybrid retrieval, two-stage pattern, metadata filtering, aggregation | `src/rag/retriever.py`, `src/rag/vector_store.py`, `src/graph/traverser.py` |
| 5 | [Grounded Generation](05-grounded-generation.md) | ~20 min | Prompt engineering for RAG, hallucination prevention, prompt templates, evidence citation | `src/core/llm_client.py` |
| 6 | [Architecture Walkthrough](06-architecture-walkthrough.md) | ~20 min | Full pipeline data flow, configuration, testing strategy, end-to-end trace | `src/analyzer/`, `src/api/`, `config/` |

**Total reading time: approximately 2 hours.**

---

## Suggested Reading Order

The chapters are sequential — each one builds on the previous. Follow them
in order from 1 through 6.

```
Chapter 1  establishes vocabulary and mental models used everywhere else.
Chapter 2  introduces the math layer (embeddings) that Chapter 4 depends on.
Chapter 3  introduces the structural layer (graphs) that Chapter 4 depends on.
Chapter 4  combines both layers into a retrieval pipeline.
Chapter 5  shows how retrieved context becomes a grounded LLM response.
Chapter 6  ties every piece together into the running application.
```

---

## Learning Path

```
                         START HERE
                             |
                             v
                  +-----------------------+
                  |  1. RAG Fundamentals  |
                  |  Why RAG exists       |
                  +-----------+-----------+
                              |
                +-------------+-------------+
                |                           |
                v                           v
   +------------------------+  +-------------------------+
   |  2. Embeddings         |  |  3. Knowledge Graphs    |
   |  Meaning as numbers    |  |  Structure as edges     |
   +------------+-----------+  +------------+------------+
                |                           |
                +-------------+-------------+
                              |
                              v
                  +-----------------------+
                  |  4. Retrieval         |
                  |  Hybrid search        |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  |  5. Grounded          |
                  |     Generation        |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  |  6. Architecture      |
                  |  Full pipeline        |
                  +-----------------------+
                              |
                              v
                          YOU DID IT
```

---

## Quick Reference

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation — fetch relevant facts, then let the LLM explain them. |
| **Embedding** | A fixed-length list of numbers that encodes the semantic meaning of text. |
| **Vector store** | A database optimized for nearest-neighbor similarity search over embeddings. |
| **Cosine similarity** | A 0-to-1 score measuring how close two embeddings are in direction. |
| **Bipartite graph** | A graph with two node types where edges only connect nodes of different types. |
| **BFS** | Breadth-First Search — traverse a graph level by level from a starting node. |
| **Hybrid retrieval** | Combining vector (semantic) search with graph (structural) traversal. |
| **Two-stage retrieval** | Cast a wide net first, then rerank and filter to keep only the best results. |
| **Grounding** | Supplying the LLM with verified facts so it explains rather than invents. |
| **Hallucination** | When an LLM generates confident but factually incorrect output. |
| **Blast radius** | The set of components transitively affected by a change to one component. |

---

## Connection to the Codebase

The docs map directly to `src/` modules. Here is how they line up:

```
docs/                           src/
--------------------------      --------------------------
01-rag-fundamentals.md     -->  (conceptual — no single module)
02-embeddings-demystified  -->  src/rag/embedder.py
                                src/rag/vector_store.py
03-knowledge-graphs        -->  src/graph/builder.py
                                src/graph/traverser.py
04-retrieval-strategies    -->  src/rag/retriever.py
05-grounded-generation     -->  src/core/llm_client.py
06-architecture-walkthrough --> src/analyzer/  (orchestration)
                                src/api/       (HTTP layer)
                                config/        (runtime settings)
```

When a chapter references a concept, it points to the exact file where that
concept is implemented. Read the chapter first, then open the source file —
the code will make sense on the first pass.

---

*Built as part of the [impact-radar](../) project.*
