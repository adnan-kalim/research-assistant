"""Retrieve-and-generate Q&A over the personal paper library.

This is stages 4 and 5 of the RAG pipeline (RETRIEVE, GENERATE) -- see
README.md. Retrieval is local and free (Qwen3-Embedding + Chroma);
generation calls Claude Haiku 4.5 once per question, grounded in whatever
was retrieved.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic

from research_assistant.config import settings
from research_assistant.library import store

_TOP_K = 6

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


def ask(question: str, top_k: int = _TOP_K) -> Answer:
    """Retrieve relevant chunks from the library and ask Claude to answer, grounded in them."""
    if not settings.anthropic_api_key:
        return Answer(
            text="No ANTHROPIC_API_KEY configured -- add one to your .env file "
            "to enable Q&A (see .env.example).",
            citations=[],
        )

    index = store.get_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)

    if not nodes:
        return Answer(
            text="Your library is empty, so there's nothing to answer from. "
            "Save some papers first with `research save`.",
            citations=[],
        )

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
