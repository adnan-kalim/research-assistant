"""Retrieve-and-generate Q&A over the personal paper library.

This is stages 4 and 5 of the RAG pipeline (RETRIEVE, GENERATE) -- see
README.md. Phase 4 adds two retrieval quality improvements on top of the
Phase 3 pure-vector baseline:

  1. Hybrid retrieval -- BM25 (keyword) and vector similarity run in
     parallel; their ranked lists are merged with Reciprocal Rank Fusion
     (RRF). Keyword search catches exact-match misses that vector similarity
     scores poorly; vector search catches semantic matches that BM25 misses.

  2. Cross-encoder reranking -- after the fused candidate pool is assembled
     (~20 chunks by default), a small cross-encoder model (ms-marco-MiniLM)
     re-scores every candidate against the *exact* query. This is much more
     accurate than the bi-encoder approximation used for retrieval. Only the
     top_n highest-scoring chunks are forwarded to Claude.

Both can be toggled off via keyword arguments (or --no-hybrid / --no-reranker
CLI flags) for A/B comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import anthropic
from llama_index.core.schema import QueryBundle

from research_assistant.config import settings
from research_assistant.library import store

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore

_WIDE_K = settings.retrieval_wide_k   # wide pool retrieved before reranking
_TOP_K = settings.reranker_top_n      # final chunks kept for generation

_SYSTEM_PROMPT = """\
You are a research assistant answering questions using ONLY the excerpts \
provided below from the user's saved papers. Each excerpt is labeled with \
the paper it came from.

Rules:
- Ground every claim in the excerpts. Cite the paper for each claim using \
  its label, e.g. "(Smith et al., 2023)".
- If the excerpts don't contain enough information to answer, say so \
  plainly instead of guessing or relying on outside knowledge.
- Be concise and direct.
"""

# Module-level reranker cache: the cross-encoder model (~80 MB for MiniLM) is
# expensive to load. We keep one instance alive for the process lifetime and
# update its top_n when the caller requests a different value.
_reranker: object | None = None


@dataclass
class Citation:
    paper_id: str
    title: str
    authors: str
    year: int | None


@dataclass
class Answer:
    text: str
    citations: list[Citation]


def _label(meta: dict) -> str:
    """A short "(Author et al., Year)"-style label for a chunk's source paper."""
    authors = (meta.get("authors") or "").strip()
    year = meta.get("year") or "n.d."
    if not authors:
        return f"Unknown, {year}"
    first_author = authors.split(",")[0].strip()
    suffix = " et al." if "," in authors else ""
    return f"{first_author}{suffix}, {year}"


def _get_reranker(top_n: int):
    """Return a cached SentenceTransformerRerank, loading the model once."""
    global _reranker
    from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

    if _reranker is None:
        _reranker = SentenceTransformerRerank(
            model=settings.reranker_model,
            top_n=top_n,
        )
    else:
        _reranker.top_n = top_n
    return _reranker


def _import_hybrid_deps():
    """Import hybrid retrieval packages, raising a clear error if not installed.

    Isolated here so tests can monkeypatch this single function instead of
    fighting with lazy-import patch paths.
    """
    try:
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.retrievers.bm25 import BM25Retriever

        return BM25Retriever, QueryFusionRetriever
    except ImportError as exc:
        raise ImportError(
            "Hybrid retrieval requires llama-index-retrievers-bm25. "
            "Run: pip install -r requirements.txt"
        ) from exc


def _retrieve_nodes(
    question: str, wide_k: int, use_hybrid: bool
) -> list[NodeWithScore]:
    """Retrieve candidate chunks using hybrid (BM25 + vector RRF) or vector only.

    With use_hybrid=True:
      - BM25Retriever tokenises and scores all stored chunks by keyword overlap.
      - VectorIndexRetriever scores by cosine similarity of Qwen3 embeddings.
      - QueryFusionRetriever merges both ranked lists via Reciprocal Rank Fusion.

    With use_hybrid=False: plain vector retrieval (Phase 3 behaviour).
    """
    index = store.get_index()

    if not use_hybrid:
        return index.as_retriever(similarity_top_k=wide_k).retrieve(question)

    all_nodes = store.get_all_text_nodes()
    if not all_nodes:
        return []

    BM25Retriever, QueryFusionRetriever = _import_hybrid_deps()

    bm25 = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=wide_k)
    vec = index.as_retriever(similarity_top_k=wide_k)

    return QueryFusionRetriever(
        retrievers=[bm25, vec],
        similarity_top_k=wide_k,
        num_queries=1,       # no LLM query expansion, just use the original
        mode="reciprocal_rerank",
        use_async=False,
        llm=None,
    ).retrieve(question)


def _rerank(
    nodes: list[NodeWithScore], question: str, top_n: int
) -> list[NodeWithScore]:
    """Re-score candidate nodes with a cross-encoder and keep the top_n."""
    reranker = _get_reranker(top_n)
    return reranker.postprocess_nodes(
        nodes, query_bundle=QueryBundle(query_str=question)
    )


def ask(
    question: str,
    top_k: int = _TOP_K,
    use_hybrid: bool = True,
    use_reranker: bool = True,
) -> Answer:
    """Retrieve relevant chunks from the library and ask Claude to answer, grounded in them.

    Args:
        question:      The natural-language question to answer.
        top_k:         Final number of chunks forwarded to Claude.
        use_hybrid:    If True, fuse BM25 + vector results with RRF before reranking.
        use_reranker:  If True, re-score the wide candidate pool with a cross-encoder.
    """
    if not settings.anthropic_api_key:
        return Answer(
            text="No ANTHROPIC_API_KEY configured -- add one to your .env file "
            "to enable Q&A (see .env.example).",
            citations=[],
        )

    wide_k = _WIDE_K if use_reranker else top_k
    nodes = _retrieve_nodes(question, wide_k=wide_k, use_hybrid=use_hybrid)

    if not nodes:
        return Answer(
            text="Your library is empty, so there's nothing to answer from. "
            "Save some papers first with `research save`.",
            citations=[],
        )

    if use_reranker:
        nodes = _rerank(nodes, question, top_n=top_k)
    else:
        nodes = nodes[:top_k]

    seen_papers: dict[str, Citation] = {}
    context_blocks: list[str] = []
    for node in nodes:
        meta = node.metadata or {}
        paper_id = meta.get("paper_id", "unknown")
        label = _label(meta)
        seen_papers[paper_id] = Citation(
            paper_id=paper_id,
            title=meta.get("title", "Untitled"),
            authors=meta.get("authors", ""),
            year=meta.get("year") or None,
        )
        context_blocks.append(f"[{label}] {node.get_content()}")

    context = "\n\n---\n\n".join(context_blocks)
    user_message = f"Excerpts:\n\n{context}\n\nQuestion: {question}"

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        response = client.messages.create(
            model=settings.reasoning_model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AuthenticationError:
        return Answer(
            text="Anthropic API authentication failed -- check your ANTHROPIC_API_KEY.",
            citations=[],
        )
    except anthropic.RateLimitError:
        return Answer(
            text="Rate limited by the Anthropic API -- wait a moment and try again.",
            citations=[],
        )
    except anthropic.APIConnectionError:
        return Answer(
            text="Could not reach the Anthropic API -- check your network connection.",
            citations=[],
        )
    except anthropic.APIStatusError as exc:
        return Answer(text=f"Anthropic API error: {exc.message}", citations=[])

    text = "".join(block.text for block in response.content if block.type == "text")
    return Answer(text=text, citations=list(seen_papers.values()))
