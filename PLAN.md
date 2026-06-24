# AI-Powered Research Assistant — Build Plan

## Context

You want to build an AI research assistant with access to "thousands (if not millions)" of papers. The repo is currently empty (placeholder `README.md` only), so this is a greenfield build.

Key decisions you've made:
- **Scope:** All of it — semantic *search/discovery*, conversational *Q&A* with citations, and *synthesis* (lit review). We'll phase it so each piece works before the next.
- **Corpus:** **Hybrid** — query live scholarly APIs (arXiv, Semantic Scholar, OpenAlex) for broad discovery across the millions of papers they already index, *plus* a **local vector store** for papers you explicitly save to a "personal library" for deep Q&A. You do **not** need to download or host millions of papers.
- **Interface:** Backend competencies first, structured so a UI can sit on top later (Phase 6 adds a thin API layer).
- **Models:** Local **Qwen3-Embedding** for embeddings + retrieval (free, runs on your machine); **Claude Haiku 4.5** (`claude-haiku-4-5`, ~$1/$5 per 1M input/output tokens) for the reasoning/synthesis step.
- **Framework:** **LlamaIndex** as the core RAG framework (it is purpose-built for retrieval/ingestion, which fits a research assistant better than LangChain). We can add LangChain later if/when we need multi-step agent orchestration.
- **Learning:** This plan teaches concepts as we go — you're new to RAG/MCP/orchestration.

### The mental model (read this first)

Every RAG ("Retrieval-Augmented Generation") system is the same 5-stage pipeline. Learn this and the rest is detail:

```
1. INGEST    grab paper text (title, abstract, full text)
2. EMBED     turn text into vectors (lists of numbers) that capture meaning  → Qwen3-Embedding
3. STORE     save vectors in a vector DB you can search by similarity        → Chroma (local)
4. RETRIEVE  embed the user's question, find the most similar chunks         → Chroma + Qwen3 reranker
5. GENERATE  hand those chunks + the question to an LLM, get a cited answer  → Claude Haiku 4.5
```

The LLM is **one swappable box at stage 5**. Today it's Haiku 4.5; switching providers later is a few lines.

---

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Standard for RAG/AI tooling |
| RAG framework | **LlamaIndex** | Retrieval/ingestion-first; least glue code for this use case |
| Embeddings | **Qwen3-Embedding** (local) | Free, private; the dedicated *embedding* variant of Qwen3 — **not** the chat model. Start with `Qwen3-Embedding-0.6B` (fast, ~1GB) and move up to `-4B`/`-8B` if quality needs it |
| Reranker (optional, Phase 4) | **Qwen3-Reranker-0.6B** | Re-scores retrieved chunks for relevance; big quality win for cheap |
| Vector DB | **Chroma** (local, persistent) | Zero-config, runs in-process, free |
| Reasoning LLM | **Claude Haiku 4.5** (`claude-haiku-4-5`) | Cheap, strong; via official `anthropic` SDK |
| Live paper APIs | arXiv, Semantic Scholar, OpenAlex | Free, no/low-key access to millions of papers |
| API layer (Phase 6) | **FastAPI** | Clean async HTTP layer a future UI calls |

How embeddings/reranker run locally: via **Ollama** (`ollama pull` the model, simplest) **or** Python `sentence-transformers`/`FlagEmbedding` loading from HuggingFace. We'll decide at Phase 2 based on what installs cleanly on your Windows machine; both are first-class in LlamaIndex.

---

## Project layout

```
research-assistant/
  .env                      # ANTHROPIC_API_KEY, optional S2_API_KEY (gitignored)
  .gitignore
  requirements.txt
  README.md
  src/research_assistant/
    __init__.py
    config.py               # loads env, model names, paths
    sources/                # Phase 1 — live API clients
      arxiv_client.py
      semantic_scholar.py
      openalex.py
      base.py               # common Paper dataclass (id, title, abstract, authors, year, url)
    library/                # Phase 2 — personal saved corpus
      store.py              # Chroma + Qwen3-Embedding index (LlamaIndex)
      ingest.py             # save a Paper -> chunk -> embed -> store
    qa/                     # Phase 3 — RAG Q&A
      engine.py             # retrieve + Haiku 4.5 answer with citations
    synthesis/              # Phase 5 — lit review across many papers
      review.py
    cli.py                  # Phase 1+ — command-line entry point (search/save/ask/review)
    api/                    # Phase 6 — FastAPI app (UI sits on this later)
      main.py
  tests/
    test_sources.py
    test_library.py
    test_qa.py
  scripts/
    smoke_test.py           # one end-to-end run you can re-execute anytime
```

---

## Phased build

### Phase 0 — Project scaffold & environment
- Create the package layout above, `requirements.txt`, `.gitignore` (ignore `.env`, `*.chroma`, `__pycache__`, vector store dir), and a real `README.md`.
- `config.py`: load `.env` via `python-dotenv`; expose `ANTHROPIC_API_KEY`, model IDs, and a `DATA_DIR` for the Chroma store.
- **Concept taught:** env/secret handling — the Anthropic key lives in `.env`, never in code or git.
- **Verify:** `python -c "from research_assistant.config import settings; print(settings)"` prints config with the key redacted.

### Phase 1 — Live discovery (proves "millions of papers")
- Implement `sources/arxiv_client.py` first (no API key needed), then `semantic_scholar.py` and `openalex.py`. Each returns a list of a shared `Paper` dataclass.
- `cli.py`: `research search "your topic"` → prints ranked titles/abstracts/links.
- **Concept taught:** this is *keyword/metadata* search against external indexes — no embeddings yet. It already gives you reach across millions of papers, which is the bulk of the "discovery" requirement.
- **Verify:** `research search "graph neural networks"` returns real, recent papers from at least arXiv.

### Phase 2 — Personal library + local vector store (the RAG foundation)
- `library/store.py`: build a LlamaIndex `VectorStoreIndex` backed by a **persistent Chroma** collection, using **Qwen3-Embedding** as the embed model.
- `library/ingest.py`: `save(paper)` → fetch abstract (and full text/PDF where freely available) → chunk → embed → upsert into Chroma. De-dupe by paper ID.
- `cli.py`: `research save <paper-id-or-search-result>` and `research library` (list saved).
- **Concepts taught:** embeddings (text→vectors), chunking (why we split long papers), vector similarity, persistence (your library survives restarts).
- **Verify:** save 3–5 papers, confirm the Chroma directory grows and `research library` lists them.

### Phase 3 — Conversational Q&A with citations (the core feature)
- `qa/engine.py`: a LlamaIndex query engine — embed the question, retrieve top-k chunks from Chroma, pass them + the question to **Claude Haiku 4.5** via the `anthropic` SDK, return an answer that **cites which saved papers** it used.
- `cli.py`: `research ask "your question"`.
- **Concepts taught:** the full retrieve→generate loop; grounding/citations (answers reference retrieved sources, reducing hallucination); prompt construction (how retrieved context is fed to the model).
- **LLM call specifics:** use `anthropic` Python SDK, `model="claude-haiku-4-5"`, `max_tokens≈4096`. (LlamaIndex has an Anthropic LLM wrapper we can use, or call the SDK directly — we'll pick the simplest that keeps the model swappable.)
- **Verify:** ask a question answerable from your saved papers; answer is correct *and* cites the right papers. Then ask something unsaved → it should say it lacks the source rather than inventing one.

### Phase 4 — Retrieval quality: reranking + hybrid retrieval
- Add **Qwen3-Reranker** as a LlamaIndex node-postprocessor: retrieve a wider top-k, rerank, keep the best few before generation.
- Optionally add keyword (BM25) + vector hybrid retrieval.
- **Concept taught:** retrieval quality dominates answer quality — better-ranked context beats a bigger model. This is the highest-leverage tuning step.
- **Verify:** A/B a few questions with and without the reranker; confirm more relevant chunks surface.

### Phase 5 — Synthesis / automated literature review
- `synthesis/review.py`: given a topic, pull many papers (live APIs + library), then have Haiku 4.5 synthesize themes, consensus, disagreements, and gaps **with per-claim citations**. Use map-reduce style summarization (summarize batches, then summarize the summaries) so we stay within context limits.
- `cli.py`: `research review "your topic"`.
- **Concepts taught:** synthesis across many documents; map-reduce summarization; managing context-window limits.
- **Verify:** `research review "<a topic you know>"` produces a structured, cited overview whose claims trace back to real papers.

### Phase 6 — Thin API layer (so a UI can attach later)
- `api/main.py`: FastAPI exposing `/search`, `/save`, `/ask`, `/review` over the same engine functions the CLI uses (no logic duplicated — both call the same core).
- **Concept taught:** separating core engine from interface — the CLI and a future web UI are just two front-ends over one backend.
- **Verify:** `uvicorn` serves the app; `/ask` returns the same cited answer as the CLI. (UI itself is out of scope for now, by your choice.)

### Phase 7 (optional, later) — MCP server
- Wrap the assistant's tools (`search`, `ask`, `review`) as an **MCP server** so Claude Desktop / Claude Code can call your research assistant as a tool.
- **Concept taught:** MCP (Model Context Protocol) — a standard way to expose tools to LLM apps. This is the cleanest way to learn MCP once the core works. Deferred until Phases 1–3 are solid.

---

## Key reuse / things not to hand-roll
- **LlamaIndex** provides the `VectorStoreIndex`, Chroma integration, embedding-model wrappers, query engine, rerank postprocessor, and map-reduce summarizer — use these instead of writing chunking/retrieval/prompt-stitching by hand.
- **Official `anthropic` SDK** for the Haiku 4.5 call — don't hand-roll HTTP. Default `max_tokens≈4096`; non-streaming is fine at this size.
- **`arxiv`, and the Semantic Scholar / OpenAlex REST APIs** for sources — don't scrape.

## First-session deliverable
Get **Phases 0 → 1 → 2 → 3** working end to end: search live → save a few papers → ask a grounded, cited question. That's a real, useful research assistant. Phases 4–7 are quality and surface-area improvements layered on top.

## Verification (end-to-end smoke test)
`scripts/smoke_test.py` runs the whole happy path so you can re-check at any time:
1. `search "<topic>"` returns live results.
2. `save` 3 of them into the Chroma-backed library.
3. `ask "<question>"` returns a correct answer citing the saved papers.
4. (After Phase 5) `review "<topic>"` returns a cited synthesis.
Run it with `python scripts/smoke_test.py`; it should complete without errors and print a cited answer.
