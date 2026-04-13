# Chapter 5: Grounded Generation — Making the LLM Trustworthy

> **Reading time**: ~20 minutes
> **Prerequisites**: Chapters 1-4
> **After this chapter**: You'll understand how to take raw LLM output from
> "plausible-sounding fiction" to "evidence-backed analysis" using prompt
> engineering, temperature control, context window management, and
> hallucination detection.
> **Project files**: `src/core/llm_client.py`, `src/analyzer/impact.py`

---

## The Generation Problem

You've spent four chapters building an incredible retrieval system. You have
graph paths, semantic matches, coupling strengths, and criticality scores.
Now you need an LLM to explain what it all means.

Here's what happens when you skip everything we've built and just ask:

**Without retrieval (raw LLM):**
```
You: What's the blast radius of changing AuthEngine in NexusSaaS?

ChatGPT: Changing AuthEngine could affect several components. Authentication
is typically central to any SaaS platform, so you'd want to check your
API gateway, user management, session handling, and any SSO integrations.
Consider also database migrations and caching layers that store auth tokens.
I'd recommend a staged rollout with feature flags...
```

Sounds reasonable, right? Except NONE of that is grounded in your actual
system. There's no SSO integration. There are no feature flags. The LLM
is pattern-matching against "what auth systems usually look like" and
generating plausible fiction.

**With retrieved context (grounded):**
```
You: [provides graph paths + semantic matches, then asks the same question]

LLM: AuthEngine changes directly impact EncryptionService via crypto_utils
(tight coupling, confirmed). WebSocketGateway shares session_store, which
means active connections will drop if session format changes (confirmed via
graph path). AuditLogger receives events via event_bus — log schema changes
are likely needed (indirect, depth=2). The Enterprise EU variant is at
highest risk because it includes all three components and compliance_engine
depends on AuditLogger for regulatory reporting.
```

Every claim maps to evidence. "Confirmed" means a direct dependency path
exists. "Likely" means indirect. No hand-waving about SSO or feature flags.

That's the difference between generation and GROUNDED generation.
The "G" in RAG isn't just about generating text — it's about generating
text that's tethered to facts your retrieval system actually found.

---

## The Grounding Pipeline

Raw retrieval results aren't ready for the LLM. They need to be shaped
into a prompt that constrains the LLM's output. Here's the pipeline:

```
┌──────────────────┐     ┌───────────────────┐     ┌────────────────────┐     ┌────────────────┐
│  Retrieved Facts │     │  Prompt Template  │     │    Constrained     │     │   Verified     │
│                  │────▶│                   │────▶│    Generation      │────▶│   Output       │
│ - graph paths    │     │ - system prompt   │     │                    │     │                │
│ - semantic hits  │     │ - labeled sections│     │ - temperature=0.2  │     │ - cross-check  │
│ - coupling types │     │ - grounding rules │     │ - evidence cited   │     │   vs graph     │
│ - scores         │     │ - task definition │     │ - uncertainty      │     │ - confidence   │
│                  │     │                   │     │   markers used     │     │   calibrated   │
└──────────────────┘     └───────────────────┘     └────────────────────┘     └────────────────┘
```

Each stage serves a specific purpose:

1. **Retrieved Facts**: The raw outputs from Chapters 3 and 4 — graph paths
   with coupling metadata, semantic matches with similarity scores.

2. **Prompt Template**: Structures the facts into labeled sections the LLM
   can reference. Separates evidence from instructions. Sets grounding rules.

3. **Constrained Generation**: The LLM generates with low temperature and
   explicit instructions to cite evidence and use uncertainty markers.

4. **Verified Output**: The generated text is cross-referenced against the
   graph data. If the LLM claims a path exists, we can verify it does.

---

## Prompt Engineering for RAG

This is the most important section in this chapter. In a RAG system, the
prompt isn't a casual question — it's a carefully structured document with
distinct sections that serve different roles.

### The System Prompt: Setting the Rules

The system prompt tells the LLM WHO it is, WHAT format to use, and HOW to
handle evidence. Here's the actual system prompt from impact-radar's
`_build_system_prompt()` method in `src/analyzer/impact.py`:

```
You are a senior platform risk analyst for NexusSaaS. Your job is to assess
the impact of component changes on product variants. Rules:
- Ground every claim in the evidence provided. Do NOT speculate.
- Use uncertainty markers: 'confirmed' for direct dependencies, 'likely'
  for indirect, 'possible' for semantic-only matches.
- Keep explanations concise — 2-4 sentences per variant.
- Highlight non-obvious risks that a developer might miss.
- Suggest specific mitigation steps when possible.
```

Look at what this accomplishes:

- **Role** ("senior platform risk analyst") — prevents the LLM from
  acting like a general assistant or creative writer.
- **Grounding constraint** ("Ground every claim... Do NOT speculate") —
  the single most important rule. Without this, the LLM will happily
  invent dependencies that don't exist.
- **Uncertainty markers** ("confirmed"/"likely"/"possible") — forces the
  LLM to distinguish between things it KNOWS from the evidence and things
  it's inferring. This is gold for the reader.
- **Output format** ("2-4 sentences") — prevents the LLM from writing
  a five-paragraph essay when a focused analysis is needed.

### The User Prompt: Facts, Then Question

The user prompt has two clearly separated concerns:

```
RETRIEVED CONTEXT:   Here are the facts we found.
QUESTION:            Now explain what they mean.
```

Why separate them? Because LLMs are susceptible to confusing instructions
with data. If you mix your question into the evidence, the LLM might treat
parts of your evidence as instructions, or parts of your instructions as
facts. Clean separation eliminates this class of errors.

In impact-radar, `_build_llm_prompt()` creates four labeled sections:
`## Change Scope`, `## Structural Evidence`, `## Semantic Evidence`, and
`## Task`. The LLM sees clear boundaries between "what changed," "what
the graph found," "what vector search found," and "what to do with it."

---

## Temperature: The Creativity Dial

Temperature controls how "creative" the LLM gets when choosing its next
word. Think of it as a dial:

```
Temperature = 0.0          Temperature = 0.5          Temperature = 1.0
    │                          │                          │
    ▼                          ▼                          ▼
  Always picks               Sometimes                 Frequently picks
  the most likely            explores                  surprising words
  next word                  alternatives
    │                          │                          │
    ▼                          ▼                          ▼
  Deterministic,             Balanced                   Creative,
  repetitive                                            unpredictable
```

Impact-radar uses **temperature = 0.2**. You can see it in two places:
the `config/model_config.yaml` file (`temperature: 0.2`) and in the
`_generate_explanations()` method which passes `temperature=0.2` directly.

Why 0.2 and not 0.0? Pure zero can cause degenerate repetition in some
models. A tiny bit of randomness (0.2) keeps the text natural while
staying firmly factual. We're writing risk analysis, not poetry.

When would you use higher temperature? If you were building a creative
writing tool (0.7-0.9), a brainstorming assistant (0.6-0.8), or a chatbot
that needs personality (0.5-0.7). But for ANY system where factual
accuracy matters — medical, legal, financial, risk analysis — keep it low.

---

## Context Window Management

LLMs have a fixed context window — the total amount of text they can
"see" at once. GPT-4o supports 128K tokens (roughly 96,000 words). That
sounds enormous, but there's a trap.

**More context is not always better.**

Imagine you're a detective. Someone hands you a 300-page document and
says "the critical clue is on page 147." You'd probably miss it. But
if they hand you a 5-page summary of the key evidence, you'd nail it.

LLMs work the same way. Research consistently shows that LLMs perform
worse when given too much context, especially for information in the
middle of long prompts (the "lost in the middle" problem).

This is why impact-radar's retrieval pipeline reranks to the **top 5**
results, not the top 50:

```
config/model_config.yaml:
  retrieval:
    top_k: 10              # Stage 1: cast a wide net (10 candidates)
    rerank_top_k: 5         # Stage 2: keep only the best 5
```

The goldilocks zone for RAG context:
- **Too little** (1-2 results): misses important impacts, LLM lacks
  evidence to make connections
- **Too much** (20+ results): noise drowns signal, LLM latches onto
  irrelevant details, costs more tokens
- **Just right** (3-7 results): enough evidence for a complete picture,
  focused enough for accurate generation

This is also why we generate one prompt per variant instead of one giant
prompt for everything. Each prompt contains only the evidence relevant
to THAT variant's affected components, keeping the context tight and
focused.

---

## Grounding Techniques

Getting the LLM to stay grounded isn't just about saying "don't
speculate." You need structural techniques that make grounding the
path of least resistance.

### Evidence Chains

Instead of giving the LLM a flat list of affected components, give it
the chain of dependencies that explains WHY each component is affected:

```
Flat list (bad):
  Affected: EncryptionService, WebSocketGateway, AuditLogger, ReportGenerator

Evidence chain (good):
  AuthEngine → crypto_utils → EncryptionService (depth=1, coupling=tight)
  AuthEngine → session_store → WebSocketGateway (depth=1, coupling=tight)
  AuthEngine → event_bus → AuditLogger (depth=1, coupling=loose)
  AuditLogger → compliance_engine → ReportGenerator (depth=2, coupling=loose)
```

The flat list invites the LLM to guess WHY these components are affected.
The evidence chain tells it exactly why — the LLM just needs to explain
the implications in plain English.

### Structured Prompts with Labeled Sections

Look at how `_build_llm_prompt()` organizes evidence:

```
## Change Scope
Components changed: AuthEngine, BillingCore
Variant under assessment: Enterprise EU (enterprise, eu)

## Structural Evidence (dependency graph)
- AuthEngine → crypto_utils → EncryptionService (depth=1, min_coupling=tight)
- AuthEngine → event_bus → AuditLogger → compliance_engine →
  ReportGenerator (depth=2)

## Semantic Evidence (vector similarity)
- EncryptionService (similarity=0.82) — "handles certificate rotation
  and key management for all encrypted data at rest..."

## Task
Explain why Enterprise EU is at risk from the changes to AuthEngine,
BillingCore. Highlight any non-obvious indirect dependencies.
Suggest mitigation steps.
```

Each `##` header creates a named zone. The LLM can reference "structural
evidence" or "semantic evidence" explicitly. It knows which evidence type
each fact came from, which lets it calibrate its confidence.

### Citing Evidence

The system prompt tells the LLM to use uncertainty markers:

| Marker        | Meaning                    | Evidence Source              |
|---------------|----------------------------|------------------------------|
| "confirmed"   | Direct dependency exists   | Graph path, depth=1          |
| "likely"      | Indirect dependency chain  | Graph path, depth>=2         |
| "possible"    | Conceptual similarity only | Semantic match, no graph path|

This mapping isn't random. It follows the coupling strength hierarchy
from Chapter 3: tight coupling means confirmed impact, loose coupling
means likely, and optional or semantic-only means possible.

When the LLM writes "EncryptionService is confirmed to be impacted via
crypto_utils," the reader knows this is backed by a real dependency path.
When it writes "SearchService is possibly impacted based on semantic
similarity," the reader knows to investigate further before acting.

---

## The Full RAG Prompt Template

Let's put it all together. Here's what a complete prompt looks like when
impact-radar analyzes changes to AuthEngine and BillingCore for the
Enterprise EU variant:

```
SYSTEM:
You are a senior platform risk analyst for NexusSaaS. Your job is to
assess the impact of component changes on product variants. Rules:
- Ground every claim in the evidence provided. Do NOT speculate.
- Use uncertainty markers: 'confirmed' for direct dependencies, 'likely'
  for indirect, 'possible' for semantic-only matches.
- Keep explanations concise — 2-4 sentences per variant.
- Highlight non-obvious risks that a developer might miss.
- Suggest specific mitigation steps when possible.

USER:
## Change Scope
Components changed: AuthEngine, BillingCore
Variant under assessment: Enterprise EU (enterprise, eu)

## Structural Evidence (dependency graph)
- AuthEngine → crypto_utils → EncryptionService (depth=1, min_coupling=tight)
- AuthEngine → session_store → WebSocketGateway (depth=1, min_coupling=tight)
- AuthEngine → event_bus → AuditLogger (depth=1, min_coupling=loose)
- BillingCore → pricing_rules → PricingEngine (depth=1, min_coupling=tight)
- AuditLogger → compliance_engine → ReportGenerator (depth=2)
  Shared modules: event_bus (coupling: loose)

## Semantic Evidence (vector similarity)
- EncryptionService (similarity=0.82) — "handles certificate rotation
  and key management for all encrypted data at rest"
- PricingEngine (similarity=0.74) — "calculates tier-specific pricing
  based on regional tax rules and currency conversion"

## Task
Explain why the Enterprise EU variant is at risk from the changes to
AuthEngine, BillingCore. Highlight any non-obvious indirect dependencies.
Suggest mitigation steps.
```

And the LLM generates something like:

```
Enterprise EU faces high risk from these changes. EncryptionService is
confirmed impacted via AuthEngine's crypto_utils dependency — certificate
rotation logic will need regression testing, especially for EU data
residency requirements. WebSocketGateway's session_store coupling
(confirmed) means active enterprise sessions will disconnect if the
session token format changes; consider a rolling migration.

The non-obvious risk: AuditLogger receives AuthEngine events via event_bus
(likely impacted), and compliance_engine depends on AuditLogger to feed
ReportGenerator. EU regulatory reports could silently break if audit event
schemas change. Mitigation: run the compliance report suite before deploy
and verify event schema backward compatibility.

PricingEngine is confirmed impacted via BillingCore's pricing_rules.
EU-specific VAT calculations and EUR currency handling should be
regression tested against the pricing_rules changes.
```

Every sentence traces back to specific evidence. The reader can verify
each claim. That's grounded generation.

---

## Hallucination Detection

Even with all these techniques, LLMs can still hallucinate. The final
safety net is cross-referencing the LLM's output against your source data.

If the LLM claims "AuthEngine depends on SearchService via full_text_index,"
you can check:
1. Does a graph path from AuthEngine to SearchService exist?
2. Is "full_text_index" a real shared module?
3. What's the actual coupling type?

If any of these checks fail, the claim is a hallucination.

Impact-radar implements this through its scoring and evidence structure.
Every `ScoredImpact` carries the actual paths and coupling metadata that
went into the prompt. Post-generation, you can compare what the LLM
said against what the data shows.

**Confidence calibration** ties directly to evidence type:

```
┌─────────────────────────────────────────────────────────────┐
│                   CONFIDENCE LEVELS                         │
│                                                             │
│  Evidence Type          │ Coupling   │ Confidence           │
│  ───────────────────────┼────────────┼───────────────────── │
│  Direct graph path      │ tight      │ "confirmed"          │
│  Direct graph path      │ loose      │ "confirmed"          │
│  Indirect graph path    │ tight      │ "likely"             │
│  Indirect graph path    │ loose      │ "likely"             │
│  Indirect graph path    │ optional   │ "possible"           │
│  Semantic match only    │ n/a        │ "possible"           │
│  No evidence            │ n/a        │ HALLUCINATION        │
└─────────────────────────────────────────────────────────────┘
```

The principle: if the LLM makes a claim and there's no corresponding
entry in the retrieved evidence, treat it as hallucination. Don't include
it in the report. The LLM is there to EXPLAIN evidence, not to
GENERATE evidence.

This is the fundamental shift in thinking that RAG requires. The LLM
is not a knowledge base. It's an explanation engine that operates
on facts you provide.

---

## Connect to Code

Everything in this chapter maps directly to two files in the project.

### `src/core/llm_client.py` — The Reliable Caller

`LLMClient` wraps the OpenAI API with config-driven parameters and
retry logic. The key design decisions:

- **Config-driven**: model, temperature, max_tokens, timeout all come
  from `config/model_config.yaml`. No hardcoded values. When you want
  to experiment with GPT-4o-mini for cost savings, change one YAML
  line — don't touch code.
- **Exponential backoff**: retries on `RateLimitError`,
  `APIConnectionError`, and `APITimeoutError` with delays of 1s, 2s,
  4s (configurable via `retry_base_delay`). Does NOT retry on
  `APIError` — that's a bug in your request, not a transient failure.
- **Clean separation**: `generate()` takes a prompt and optional
  system_prompt. It doesn't know about graphs or semantic matches.
  That's the orchestrator's job.

### `src/analyzer/impact.py` — The Orchestrator

`ImpactAnalyzer` is where everything from this chapter lives:

- `_build_system_prompt()` — returns the system prompt with role,
  grounding rules, and output format constraints (lines 484-508).
- `_build_llm_prompt()` — constructs the four-section user prompt
  with Change Scope, Structural Evidence, Semantic Evidence, and Task
  (lines 510-586).
- `_generate_explanations()` — calls `self._llm.generate()` with
  `temperature=0.2` for each variant, with graceful degradation if the
  LLM call fails (lines 444-482).

The graceful degradation is worth noting: if the LLM API is down, the
analysis still completes with scores, paths, and evidence — just without
the natural-language explanation. The LLM adds value but isn't a single
point of failure. Your retrieval pipeline's structured data stands on
its own.

---

## Key Takeaways

1. **Raw LLMs hallucinate.** Without retrieved context, they generate
   plausible-sounding fiction based on training patterns.

2. **The prompt is a contract.** System prompt sets the role and rules.
   User prompt separates evidence from instructions. Labeled sections
   prevent the LLM from confusing data with directives.

3. **Low temperature for facts.** Temperature 0.2 keeps generation
   deterministic and factual. Save high temperatures for creative tasks.

4. **Less context is often more.** Reranking to top 5 beats dumping
   top 50 into the prompt. LLMs lose information in long contexts.

5. **Evidence chains beat flat lists.** Showing the dependency path
   (A -> B -> C) gives the LLM the "why," not just the "what."

6. **Uncertainty markers build trust.** "Confirmed," "likely," and
   "possible" let readers calibrate how much to trust each claim.

7. **Cross-reference everything.** If the LLM claims a dependency
   exists, verify it against the graph. No evidence = hallucination.

---

**Next chapter**: [Chapter 6 — Reporting and Output Formats](06-reporting.md),
where we turn all this analysis into actionable reports for different
audiences.
