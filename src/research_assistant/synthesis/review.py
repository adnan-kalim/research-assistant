"""Automated literature review via map-reduce over multiple papers.

Given a topic, this module:

  1. GATHER  -- pulls papers from live APIs (arXiv, Semantic Scholar, OpenAlex)
               and from the saved personal library, deduplicating by paper ID.

  2. MAP     -- calls Claude once per batch of papers (~8 at a time) to produce
               a focused 2-3 sentence summary of each paper's contribution
               *relative to the topic*. Batching keeps the number of API calls
               low while avoiding prompts that are too long to parse reliably.

  3. REDUCE  -- passes all per-paper summaries to Claude in a single call and
               asks for a structured synthesis: main themes, consensus,
               disagreements, and gaps -- every claim cited back to a paper.

The two-stage design is necessary because dumping 20+ full abstracts into one
prompt produces an unfocused synthesis. Summarizing first (MAP) forces Claude
to distil each paper's relevance, giving the REDUCE step tighter, more
citable material to work with. It also keeps us within context limits if the
corpus grows large: just add more MAP batches before the single REDUCE call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import anthropic

from research_assistant.config import settings
from research_assistant.library import store
from research_assistant.sources.aggregator import search_all
from research_assistant.sources.query_rewriter import rewrite as _rewrite

_MAP_BATCH_SIZE = 8       # papers per MAP API call
_MAX_TEXT_CHARS = 2_000   # truncate very long abstracts/chunks before sending

_MAP_SYSTEM = """\
You are a research analyst. You will receive a batch of paper abstracts and a \
research topic. For each paper write a focused 2-3 sentence summary of its key \
contribution *as it relates to the topic*. Be specific and factual.

Format your response exactly like this — one block per paper, labelled with \
its number, nothing else:

[1] <summary>

[2] <summary>

...\
"""

_REDUCE_SYSTEM = """\
You are a research synthesizer writing a concise, structured literature review. \
You will receive focused summaries of multiple papers on a topic. Produce a \
synthesis with these four clearly headed sections:

## Main Themes
What ideas and approaches recur across the papers?

## Consensus
What do most papers agree on?

## Disagreements & Open Questions
Where do the papers conflict, or what remains unsettled?

## Gaps
What does the literature not yet address?

Ground every claim in the provided summaries. Cite papers inline using their \
label, e.g. "Smith et al., 2023". Do not invent claims beyond what the \
summaries say.\
"""


@dataclass
class PaperSummary:
    paper_id: str
    label: str    # "First Author et al., Year" — used for inline citations
    title: str
    source: str
    summary: str  # MAP output


@dataclass
class LitReview:
    topic: str
    text: str                  # REDUCE output: the full structured synthesis
    sources: list[PaperSummary]

    @property
    def paper_count(self) -> int:
        return len(self.sources)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _RawPaper:
    """Normalised paper representation used internally during gather + MAP."""
    paper_id: str
    title: str
    authors: str
    year: int | None
    source: str
    text: str   # title + abstract (or stored chunk text for library papers)


def _make_label(authors: str, year: int | None) -> str:
    """Return a short citation label, e.g. "Smith et al., 2023"."""
    if not authors:
        return f"Unknown, {year or 'n.d.'}"
    first = authors.split(",")[0].strip()
    suffix = " et al." if "," in authors else ""
    return f"{first}{suffix}, {year or 'n.d.'}"


def _gather_papers(
    topic: str,
    live_limit: int,
    include_live: bool,
    source_names: list[str] | None = None,
    use_rewriter: bool = False,
    queries: list[str] | None = None,
) -> list[_RawPaper]:
    """Collect papers from live APIs and the saved library, deduplicated.

    Papers that appear in both sources (you saved a paper that also shows up
    in live search results) are kept only once, with the live-API version
    taking precedence (it has a fresh abstract).

    Args:
        source_names:  Which live sources to query. None means all sources.
                       Resolved by domains.resolve_sources() before this is called.
        use_rewriter:  If True, expand the topic into 2-3 keyword queries via
                       the local rewriter model before hitting APIs (better
                       recall for natural-language topics).
        queries:       Pre-computed rewritten queries. If provided, used directly
                       and use_rewriter is ignored. Lets the CLI call _rewrite
                       once, print the results, and pass them here.
    """
    papers: dict[str, _RawPaper] = {}

    if include_live:
        if queries is not None:
            live_queries = queries
        elif use_rewriter:
            live_queries = _rewrite(topic)
        else:
            live_queries = [topic]
        for q in live_queries:
            live, _ = search_all(q, limit=live_limit, source_names=source_names)
            for p in live:
                if p.id in papers:
                    continue   # first query's version wins (usually best match)
                text = f"{p.title}\n\n{p.abstract}" if p.abstract else p.title
                papers[p.id] = _RawPaper(
                    paper_id=p.id,
                    title=p.title,
                    authors=", ".join(p.authors),
                    year=p.year,
                    source=p.source,
                    text=text,
                )

    library_texts = store.get_paper_texts()
    for meta in store.list_saved_papers():
        pid = meta.get("paper_id", "")
        if not pid or pid in papers:
            continue
        papers[pid] = _RawPaper(
            paper_id=pid,
            title=meta.get("title", "Untitled"),
            authors=meta.get("authors", ""),
            year=meta.get("year") or None,
            source=meta.get("source", "library"),
            text=library_texts.get(pid, meta.get("title", "")),
        )

    return list(papers.values())


def _map_batch(
    batch: list[_RawPaper], topic: str, client: anthropic.Anthropic
) -> list[str]:
    """Summarise a batch of papers w.r.t. topic. Returns one summary string per paper."""
    paper_blocks: list[str] = []
    for i, p in enumerate(batch, 1):
        text = p.text[:_MAX_TEXT_CHARS] if len(p.text) > _MAX_TEXT_CHARS else p.text
        label = _make_label(p.authors, p.year)
        paper_blocks.append(f"[{i}] {label}: {p.title}\n{text}")

    user_msg = f"Topic: {topic}\n\nPapers:\n\n" + "\n\n".join(paper_blocks)

    resp = client.messages.create(
        model=settings.reasoning_model,
        max_tokens=1024,
        system=_MAP_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")

    # Parse "[N] summary" blocks produced by the MAP prompt.
    summaries = [""] * len(batch)
    for m in re.finditer(r"\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)", raw, re.DOTALL):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(batch):
            summaries[idx] = m.group(2).strip()
    return summaries


def _reduce(
    paper_summaries: list[PaperSummary], topic: str, client: anthropic.Anthropic
) -> str:
    """Synthesise all per-paper summaries into one structured literature review."""
    blocks = [
        f"[{ps.label}] {ps.title}\n{ps.summary}" for ps in paper_summaries
    ]
    user_msg = (
        f"Topic: {topic}\n\n"
        f"Paper summaries ({len(paper_summaries)} papers):\n\n"
        + "\n\n---\n\n".join(blocks)
    )

    resp = client.messages.create(
        model=settings.reasoning_model,
        max_tokens=2048,
        system=_REDUCE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def review(
    topic: str,
    live_limit: int = 10,
    include_live: bool = True,
    source_names: list[str] | None = None,
    use_rewriter: bool = True,
    queries: list[str] | None = None,
) -> LitReview:
    """Run a map-reduce literature review on `topic`.

    Args:
        topic:         The research topic to synthesise.
        live_limit:    Papers to fetch per live API source (arXiv, S2, OpenAlex).
        include_live:  If False, only the saved library is used (no API calls).
        source_names:  Which live sources to query (resolved by the CLI via
                       domains.resolve_sources). None means all sources.
        use_rewriter:  If True (default), expand the topic into 2-3 keyword
                       queries before hitting APIs. Pass False or --no-rewrite
                       to skip (useful if you've already crafted tight terms).

    Returns a LitReview with the synthesis text and the full source list.
    """
    if not settings.anthropic_api_key:
        return LitReview(
            topic=topic,
            text="No ANTHROPIC_API_KEY configured -- add one to your .env file.",
            sources=[],
        )

    raw_papers = _gather_papers(
        topic,
        live_limit=live_limit,
        include_live=include_live,
        source_names=source_names,
        use_rewriter=use_rewriter,
        queries=queries,
    )
    if not raw_papers:
        return LitReview(
            topic=topic,
            text=(
                "No papers found for this topic. Try `research search` first, "
                "save some papers, or broaden the topic."
            ),
            sources=[],
        )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # MAP: summarise in batches of _MAP_BATCH_SIZE
    paper_summaries: list[PaperSummary] = []
    for batch_start in range(0, len(raw_papers), _MAP_BATCH_SIZE):
        batch = raw_papers[batch_start : batch_start + _MAP_BATCH_SIZE]
        try:
            summaries = _map_batch(batch, topic, client)
        except anthropic.APIError as exc:
            return LitReview(
                topic=topic,
                text=f"Anthropic API error during MAP step: {exc}",
                sources=[],
            )
        for p, summary in zip(batch, summaries):
            paper_summaries.append(
                PaperSummary(
                    paper_id=p.paper_id,
                    label=_make_label(p.authors, p.year),
                    title=p.title,
                    source=p.source,
                    summary=summary or f"(No summary extracted for: {p.title})",
                )
            )

    # REDUCE: synthesise all summaries into a structured review
    try:
        synthesis = _reduce(paper_summaries, topic, client)
    except anthropic.APIError as exc:
        return LitReview(
            topic=topic,
            text=f"Anthropic API error during REDUCE step: {exc}",
            sources=paper_summaries,
        )

    return LitReview(topic=topic, text=synthesis, sources=paper_summaries)
