# Documentation

Full reference for every service, CLI command, and configuration option in the research assistant.

---

## Table of contents

1. [Architecture overview](#1-architecture-overview)
2. [Configuration service](#2-configuration-service)
3. [Sources — live paper discovery](#3-sources--live-paper-discovery)
4. [Domain routing](#4-domain-routing--sourcesdomainspy)
5. [Query rewriting](#5-query-rewriting--sourcesquery_rewriterpy)
6. [Library — personal corpus](#6-library--personal-corpus)
7. [Q&A engine — retrieval and generation](#7-qa-engine--retrieval-and-generation)
8. [Synthesis — literature review](#8-synthesis--literature-review)
9. [CLI reference](#9-cli-reference)
10. [Configuration reference](#10-configuration-reference)
11. [Project layout](#11-project-layout)
12. [Tech stack](#12-tech-stack)

---

## 1. Architecture overview

Every user-facing operation runs through the same five-stage RAG pipeline:

```
1. INGEST    Grab paper text (title + abstract, or full PDF)
2. EMBED     Turn text into vectors that capture meaning        → Qwen3-Embedding-0.6B (local)
3. STORE     Save vectors in a searchable vector DB             → Chroma (local, persistent)
4. RETRIEVE  Embed the question, find the best matching chunks  → BM25 + vector → RRF → reranker
5. GENERATE  Hand chunks + the question to an LLM              → Claude Haiku 4.5 (API)
```

Stages 1–4 are entirely local and free. Stage 5 calls the Anthropic API and costs roughly $0.001–$0.01 per query at Haiku 4.5 pricing.

### Phase 4 — hybrid retrieval pipeline

Pure vector search misses exact-term queries ("BERT", "GPT-4"). BM25 misses semantic queries ("making models smaller" ≠ "parameter-efficient fine-tuning"). Combining both and reranking covers both failure modes:

```
Question
  → BM25Retriever        (top-20 by keyword overlap)  ─┐
  → VectorIndexRetriever (top-20 by embedding cosine) ─┤  Reciprocal Rank Fusion
                                                        ↓
                              Cross-encoder reranker (keeps top-6)
                                                        ↓
                                               Claude Haiku 4.5
```

**Why two stages?** The embedding model (bi-encoder) encodes query and document separately — fast but approximate. The cross-encoder sees query and document *together* and scores relevance much more precisely, but is too slow to run over thousands of chunks. Running it only on the top-20 fused candidates gets the best of both.

### Phase 5 — map-reduce synthesis

Dumping 20+ abstracts into one prompt produces unfocused synthesis. Instead:

```
REWRITE Topic → 2-3 keyword queries via local Qwen2.5-0.5B-Instruct
GATHER  Live APIs (per rewritten query) + library → deduplicated paper list
MAP     Claude summarises each paper's relevance to the topic (8 per batch, one API call per batch)
REDUCE  Claude synthesises all summaries → structured review with per-claim citations
```

---

## 2. Configuration service

**File:** `src/research_assistant/config.py`

Loads all settings from the `.env` file and environment variables into a single `Settings` object. Import `settings` from this module anywhere you need configuration — never read `os.environ` directly elsewhere.

```python
from research_assistant.config import settings

settings.anthropic_api_key      # str | None
settings.reasoning_model        # "claude-haiku-4-5"
settings.embedding_model        # "Qwen/Qwen3-Embedding-0.6B"
settings.query_rewriter_model   # "Qwen/Qwen2.5-0.5B-Instruct"
settings.reranker_model         # "cross-encoder/ms-marco-MiniLM-L-6-v2"
settings.reranker_top_n         # 6
settings.retrieval_wide_k       # 20
settings.chroma_dir             # Path  (data/chroma)
settings.collection_name        # "papers"
```

`settings.redacted()` returns a dict with secrets masked — safe to print or log.

---

## 3. Sources — live paper discovery

**Directory:** `src/research_assistant/sources/`

### `base.py` — shared data model

**`Paper`** — normalised representation of a paper regardless of which API it came from.

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Stable ID, e.g. `"arxiv:2301.12345"` or `"s2:abcd1234"` |
| `source` | `str` | `"arxiv"` \| `"semantic_scholar"` \| `"openalex"` |
| `title` | `str` | Paper title |
| `abstract` | `str \| None` | Abstract text |
| `authors` | `list[str]` | Author names |
| `year` | `int \| None` | Publication year |
| `url` | `str \| None` | Human-readable landing page |
| `pdf_url` | `str \| None` | Direct PDF link, if freely available |

Notable methods:
- `paper.display()` — formatted multi-line string for CLI output
- `paper.short_authors(max_authors=3)` — "Smith, Jones et al."

**`PaperSource`** — abstract base class every source client implements. One required method: `search(query, limit) -> list[Paper]`.

---

### `arxiv_client.py` — arXiv

No API key required. Queries the arXiv API via the `arxiv` Python package. Returns papers sorted by relevance.

### `semantic_scholar.py` — Semantic Scholar

No key required at low rate limits. Set `S2_API_KEY` in `.env` for higher throughput. Returns papers with citation counts (useful for ranking).

### `openalex.py` — OpenAlex

No key required. Set `CONTACT_EMAIL` in `.env` to join OpenAlex's "polite pool" for faster, less-throttled responses.

---

### `aggregator.py` — fan-out search

```python
from research_assistant.sources.aggregator import search_all

papers, warnings = search_all("graph neural networks", limit=5)
papers, warnings = search_all("BERT", limit=10, source_names=["arxiv", "semantic_scholar"])
```

`search_all` fans the query out to all (or named) sources in parallel. A source that fails (network error, rate limit) is skipped and its error appended to `warnings` — one flaky API never aborts the whole search.

**Returns:** `(list[Paper], list[str])` — papers and any warning messages.

---

## 4. Domain routing — `sources/domains.py`

Routes search queries to the sources most likely to have relevant papers. The core decision is: **include arXiv or not**. arXiv is a STEM-only preprint server — sending a psychology or social science query there returns astrophysics papers that happen to contain the word "attention" or "ten years".

### How auto-detect works

Domain classification runs entirely locally using the already-loaded Qwen3-Embedding model:

1. Each domain has a short keyword-rich description (defined once in `DOMAIN_DESCRIPTIONS`).
2. On first use those descriptions are embedded with `store.get_embed_model()` and cached for the process lifetime.
3. At query time: embed the topic → cosine similarity against each cached domain vector → return the nearest domain → map to that domain's source list.

No API key, no network call, no extra cost. If the embedding model is already cached on disk (because you've saved at least one paper), classification adds only a few milliseconds.

### Domain table

| Domain | Sources queried | Typical topics |
|---|---|---|
| `cs` | arxiv, semantic_scholar | Machine learning, algorithms, software engineering, AI |
| `physics` | arxiv, semantic_scholar, openalex | Quantum mechanics, astrophysics, particle physics |
| `math` | arxiv, semantic_scholar, openalex | Pure mathematics, statistics, probability |
| `economics` | arxiv, semantic_scholar, openalex | Econometrics, finance, markets, policy |
| `social_science` | semantic_scholar, openalex | Sociology, political science, public policy |
| `psychology` | semantic_scholar, openalex | Cognitive science, mental health, behavior, adolescents |
| `medicine` | semantic_scholar, openalex | Clinical research, disease, pharmacology, health |
| `biology` | semantic_scholar, openalex | Genetics, ecology, molecular biology |
| `humanities` | semantic_scholar, openalex | History, philosophy, literature, linguistics |
| `all` | arxiv, semantic_scholar, openalex | Manual escape hatch — never auto-detected |

### Public API

```python
from research_assistant.sources.domains import classify_domain, resolve_sources

domain = classify_domain("attention spans among teenagers")
# → "psychology"

source_names, explanation = resolve_sources(
    domain="auto",          # or a preset, or None
    sources=None,           # or "arxiv,openalex"
    topic="attention spans among teenagers",
)
# source_names → ["semantic_scholar", "openalex"]
# explanation  → "auto-detected domain: psychology (arxiv excluded)"
```

`resolve_sources` priority chain:
1. Explicit `sources` string → used as-is, no classification.
2. Explicit `domain` preset → mapped to source list, no classification.
3. `domain == "auto"` or `None` → `classify_domain()` → mapped to source list.

---

## 5. Query rewriting — `sources/query_rewriter.py`

Live paper APIs (arXiv, Semantic Scholar, OpenAlex) use keyword matching, not semantic search. A natural-language question like *"attention spans among teenagers in the last ten years"* causes false matches on "attention" (machine-learning attention mechanisms), "teenagers" (any adolescent study), and "ten years" (any longitudinal work). Query rewriting converts the topic into 2-3 tight academic keyword phrases before hitting the APIs.

### How it works

1. The topic is sent to `Qwen/Qwen2.5-0.5B-Instruct`, a ~500 MB instruction-tuned model running locally on CPU.
2. A system prompt asks for exactly 3 short keyword queries (3-6 words each), one per line, no extra text.
3. The output is parsed: numbering (`1.`), bullets (`-`, `•`), and quotes are stripped.
4. Each rewritten query is sent to `search_all` independently; results are merged and deduplicated by paper id.
5. If the model fails for any reason, the original topic is used as-is — no error surfaces to the caller.

### Model and caching

The model is lazy-loaded on first use and cached for the process lifetime (same pattern as the embedding model). It downloads once to `C:\Users\<you>\.cache\huggingface\hub\` and is shared with any other project on the machine. Setting `device=-1` pins it to CPU so it never competes with the embedding model for GPU memory.


### Public API

```python
from research_assistant.sources.query_rewriter import rewrite

queries = rewrite("attention spans among teenagers in the last ten years")
# → ["adolescent sustained attention span",
#    "teenage focus digital media screen time",
#    "youth cognitive performance technology"]
```

`rewrite` always returns a non-empty list. On failure it returns `[topic]`.

---

## 6. Library — personal corpus

**Directory:** `src/research_assistant/library/`

### `ingest.py` — save a paper

```python
from research_assistant.library.ingest import save_paper

chunks_written = save_paper(paper)                  # title + abstract (fast, default)
chunks_written = save_paper(paper, full_text=True)  # + full PDF (slower, richer Q&A)
```

**What happens internally:**
1. Deletes any existing chunks for `paper.id` (so re-saving never duplicates).
2. Builds a text body: always `title + abstract`; optionally fetches and extracts the full PDF via `pypdf`.
3. Splits into overlapping chunks (`chunk_size=512`, `overlap=64` tokens) using LlamaIndex's `SentenceSplitter`.
4. Embeds chunks with the local Qwen3-Embedding model and upserts into Chroma.

Returns the number of chunks written. A typical abstract-only save produces 1–3 chunks; a full-text save can produce 50+.

---

### `store.py` — vector index and Chroma interface

All access to the Chroma vector store goes through this module. All heavy objects (Chroma client, embedding model, index) are lazily loaded and cached for the process lifetime.

```python
from research_assistant.library import store

index   = store.get_index()           # LlamaIndex VectorStoreIndex (lazy)
model   = store.get_embed_model()     # HuggingFaceEmbedding (lazy, downloads on first use)
n       = store.chunk_count()         # total chunks stored
papers  = store.list_saved_papers()   # list of metadata dicts, one per distinct paper
nodes   = store.get_all_text_nodes()  # all chunks as TextNode objects (used by BM25)
texts   = store.get_paper_texts()     # {paper_id: full_text} (used by synthesis)

store.delete_paper("arxiv:2301.12345")  # remove all chunks for a paper
```

`get_all_text_nodes()` and `get_paper_texts()` are Phase 4/5 additions. They materialise all stored chunks from Chroma into memory — cheap for a personal library of tens to hundreds of papers, but would need rethinking at much larger scale.

---

## 7. Q&A engine — retrieval and generation

**File:** `src/research_assistant/qa/engine.py`

### Public API

```python
from research_assistant.qa.engine import ask, Answer, Citation

answer = ask("What are the trade-offs of mixture-of-experts models?")
answer = ask("...", top_k=8)                                 # more chunks to Claude
answer = ask("...", use_hybrid=False)                        # vector-only retrieval
answer = ask("...", use_reranker=False)                      # skip cross-encoder
answer = ask("...", use_hybrid=False, use_reranker=False)    # Phase 3 baseline

answer.text        # str — the cited answer from Claude
answer.citations   # list[Citation]
```

**`Citation`** fields: `paper_id`, `title`, `authors`, `year`.

### Internal pipeline

| Function | Role |
|---|---|
| `_retrieve_nodes(question, wide_k, use_hybrid)` | Runs BM25 + vector retrieval and fuses results with Reciprocal Rank Fusion, or falls back to vector-only. |
| `_import_hybrid_deps()` | Lazily imports `BM25Retriever` and `QueryFusionRetriever`; isolated so tests can monkeypatch it without fighting lazy-import paths. |
| `_rerank(nodes, question, top_n)` | Runs the cached cross-encoder reranker and returns the top_n nodes. |
| `_get_reranker(top_n)` | Returns (or creates) the cached `SentenceTransformerRerank` instance. The cross-encoder model is loaded once and reused across calls. |
| `_label(meta)` | Formats a chunk's metadata into a `"Smith et al., 2023"` citation label. |

### Retrieval parameters

| Parameter | Default | Source |
|---|---|---|
| Wide candidate pool | 20 | `RETRIEVAL_WIDE_K` env var |
| Final chunks to Claude | 6 | `RERANKER_TOP_N` env var |
| Reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `RERANKER_MODEL` env var |

---

## 8. Synthesis — literature review

**File:** `src/research_assistant/synthesis/review.py`

### Public API

```python
from research_assistant.synthesis.review import review, LitReview, PaperSummary

lit = review("parameter-efficient fine-tuning")
lit = review("topic", live_limit=5)         # fewer live results per source
lit = review("topic", include_live=False)   # library only, no API calls

lit.topic        # str
lit.text         # str — the full four-section synthesis
lit.sources      # list[PaperSummary]
lit.paper_count  # int
```

**`PaperSummary`** fields: `paper_id`, `label` (citation string), `title`, `source`, `summary` (the MAP output for this paper).

### Internal pipeline

| Function | Role |
|---|---|
| `_gather_papers(topic, live_limit, include_live, source_names, use_rewriter, queries)` | If `queries` is provided (pre-computed by the CLI), uses them directly. Otherwise calls `_rewrite` when `use_rewriter=True`. Loops `search_all` over each query, deduplicates by `paper_id`, then merges the saved library. Live version wins on duplicates. |
| `_map_batch(batch, topic, client)` | Sends up to 8 papers to Claude with a structured prompt asking for numbered summaries (`[1]`, `[2]`, ...). Parses the response with a regex and returns one summary string per paper. |
| `_reduce(paper_summaries, topic, client)` | Sends all summaries to Claude and requests a four-section synthesis: Main Themes, Consensus, Disagreements & Open Questions, Gaps. |
| `_make_label(authors, year)` | Formats `"Smith et al., 2023"` from raw metadata. |

### MAP prompt design

The MAP prompt asks Claude to output:
```
[1] <2-3 sentence summary of paper 1's contribution to the topic>

[2] <2-3 sentence summary of paper 2's contribution to the topic>
```

Forcing numbered blocks makes parsing deterministic. If Claude omits a block, that paper gets a fallback message rather than silently disappearing from the source list.

### Scaling behaviour

- **Under ~100 papers:** the entire REDUCE step fits in one Claude context window (Haiku 4.5 has 200K tokens; 100 summaries × ~150 tokens = 15K tokens).
- **Over ~100 papers:** add an intermediate REDUCE level (summarise summaries in batches, then reduce again). Not implemented yet — not needed at personal-library scale.

---

## 9. CLI reference

All subcommands are thin wrappers in `src/research_assistant/cli.py` that delegate to the core modules above. No business logic lives in the CLI.

### `research search`

Search live academic sources.

```sh
research search "graph neural networks"
research search "attention spans among teenagers"       # auto-detects psychology → S2 + OpenAlex
research search "BERT fine-tuning" --domain cs          # preset: arxiv + semantic_scholar
research search "philosophy of mind" --domain all       # force all sources, no auto-detect
research search "X" --sources openalex                  # explicit override
```

| Flag | Default | Description |
|---|---|---|
| `--limit N` | `5` | Max results per source |
| `--domain NAME` | `auto` | Topic domain for source routing. `auto` classifies locally with the embedding model. `all` queries every source. See domain table below. |
| `--sources LIST` | — | Comma-separated explicit sources (`arxiv`, `semantic_scholar`, `openalex`). Overrides `--domain`. |
| `--no-rewrite` | off | Send the query verbatim to APIs, skipping local query rewriting. Useful when you've already written tight keyword phrases. |

Results are numbered. Pass a number to `research save`.

---

### `research save`

Embed a paper from the last search into the local library.

```sh
research save 2
research save arxiv:2301.12345
research save 3 --full-text
```

| Flag | Default | Description |
|---|---|---|
| `--full-text` | off | Also download and embed the full PDF. Much richer Q&A material, but takes several minutes per paper on CPU. |

Safe to re-run — existing chunks for that paper are deleted first.

---

### `research library`

List all saved papers.

```sh
research library
```

Prints title, authors, year, source, URL, and total chunk count across all papers.

---

### `research ask`

Ask a question grounded in your saved library.

```sh
research ask "What are the main approaches to efficient transformers?"
research ask "How does LoRA reduce parameters?" --top-k 8
research ask "..." --no-reranker      # skip cross-encoder (faster, slightly lower quality)
research ask "..." --no-hybrid        # vector-only retrieval
research ask "..." --no-hybrid --no-reranker   # Phase 3 baseline, useful for A/B comparison
```

| Flag | Default | Description |
|---|---|---|
| `--top-k N` | `6` | Final chunks forwarded to Claude |
| `--no-hybrid` | off | Vector-only retrieval, skip BM25 fusion |
| `--no-reranker` | off | Skip cross-encoder reranking |

---

### `research review`

Generate a structured literature review.

```sh
research review "parameter-efficient fine-tuning"
research review "attention spans among teenagers"       # auto-detects psychology → S2 + OpenAlex
research review "diffusion models" --domain cs          # preset: arxiv + semantic_scholar
research review "topic" --domain all --live-limit 5    # all sources, fewer results per source
research review "my specific topic" --no-live           # library only, no routing needed
research review "X" --sources semantic_scholar          # explicit override
```

| Flag | Default | Description |
|---|---|---|
| `--live-limit N` | `10` | Papers per live source |
| `--no-live` | off | Only synthesise saved library; skips routing entirely |
| `--domain NAME` | `auto` | Topic domain for source routing. `auto` classifies locally. `all` queries every source. See domain table below. |
| `--sources LIST` | — | Comma-separated explicit sources. Overrides `--domain`. |
| `--no-rewrite` | off | Send the topic verbatim to APIs, skipping local query rewriting. |

Output: four-section synthesis (Main Themes, Consensus, Disagreements & Open Questions, Gaps) followed by a numbered source list.

---

## 10. Configuration reference

Set these in your `.env` file. Environment variables always override `.env` values.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required for Q&A and review)* | Anthropic API key |
| `HF_TOKEN` | *(optional)* | HuggingFace access token. Suppresses the "unauthenticated" warning and gives higher download rate limits. Free at huggingface.co → Settings → Access Tokens. |
| `REASONING_MODEL` | `claude-haiku-4-5` | Claude model for Q&A and synthesis |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Local HuggingFace embedding model. Upgrade to `Qwen/Qwen3-Embedding-4B` for better quality at ~4× the size. |
| `QUERY_REWRITER_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Local generative model for query rewriting (~500 MB, CPU-only). Any HuggingFace instruction-tuned chat model works. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker. `Qwen/Qwen3-Reranker-0.6B` is an alternative. |
| `RERANKER_TOP_N` | `6` | Chunks kept after reranking, forwarded to Claude |
| `RETRIEVAL_WIDE_K` | `20` | Candidate pool fetched before reranking (higher = better recall, slower rerank) |
| `DATA_DIR` | `./data` | Directory where Chroma persists the vector store |
| `CHROMA_COLLECTION` | `papers` | Chroma collection name |
| `S2_API_KEY` | *(optional)* | Semantic Scholar API key for higher rate limits |
| `CONTACT_EMAIL` | *(optional)* | Email for OpenAlex's polite pool (faster responses) |

---

## 11. Project layout

```
research-assistant/
  .env                          API keys and config overrides (gitignored)
  .env.example                  Template — copy to .env and fill in
  requirements.txt              All dependencies with version pins
  README.md                     Setup guide
  DOCS.md                       This file
  PLAN.md                       Phased build plan with architecture decisions

  src/research_assistant/
    config.py                   Loads .env; exposes single `settings` object
    cli.py                      CLI entry point — thin wrappers, no logic
    session.py                  Persists last search so `save <n>` works

    sources/
      base.py                   Paper dataclass + PaperSource ABC
      arxiv_client.py           arXiv API (no key needed)
      semantic_scholar.py       Semantic Scholar API
      openalex.py               OpenAlex API
      aggregator.py             Fan-out search across all sources
      domains.py                Local embedding-based domain routing
      query_rewriter.py         Local generative query rewriting (Qwen2.5-0.5B-Instruct)

    library/
      ingest.py                 chunk → embed → upsert pipeline
      store.py                  Chroma + Qwen3-Embedding index; all store helpers

    qa/
      engine.py                 BM25 + vector → RRF → reranker → Claude cited answer

    synthesis/
      review.py                 gather → MAP → REDUCE literature review

    api/
      main.py                   FastAPI layer (Phase 6, planned)

  tests/
    test_sources.py             Source client tests (mocked HTTP)
    test_library.py             Ingest logic tests (mocked store)
    test_qa.py                  Q&A label helper tests
    test_retrieval.py           Phase 4: hybrid retrieval + reranker wiring
    test_synthesis.py           Phase 5: gather, MAP, REDUCE, end-to-end flow
    test_domains.py             Domain routing: classify, resolve, priority chain
    test_query_rewriter.py      Query rewriting: output parsing, stripping, fallback

  scripts/
    smoke_test.py               End-to-end happy path: search → save → ask
```

---

## 12. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Standard for RAG/AI tooling |
| RAG framework | LlamaIndex | Retrieval/ingestion-first; least glue code for this use case |
| Embeddings | Qwen3-Embedding-0.6B | Free, local, private. The dedicated embedding variant — not the Qwen3 chat model. |
| Query rewriter | Qwen2.5-0.5B-Instruct | ~500 MB local generative model; converts natural-language topics into tight keyword queries before hitting APIs. |
| Reranker | ms-marco-MiniLM-L-6-v2 | ~80 MB cross-encoder; big quality win, runs fast on CPU |
| Vector DB | Chroma | Zero-config, runs in-process, fully persistent |
| Hybrid retrieval | BM25 + vector → RRF | Covers each other's failure modes; RRF fuses rank lists without comparing incompatible raw scores |
| Reasoning LLM | Claude Haiku 4.5 | Cheap and strong; ~$1/$5 per 1M in/out tokens; swappable via `REASONING_MODEL` |
| Live paper APIs | arXiv, Semantic Scholar, OpenAlex | Free; combined reach of 200M+ papers |
| API layer | FastAPI (Phase 6) | Thin HTTP surface; CLI and future UI share the same core modules |
