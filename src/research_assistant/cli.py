"""Command-line entry point for the research assistant.

    research search "graph neural networks"
    research save 2
    research library
    research ask "..."
    research review "..."

Each subcommand is a thin wrapper around the corresponding module --
the CLI itself never contains business logic, so the same logic is
reusable from the FastAPI layer (Phase 6) without duplication.
"""

from __future__ import annotations

import argparse
import sys

from research_assistant import session
from research_assistant.library import ingest, store
from research_assistant.qa import engine as qa_engine
from research_assistant.sources.aggregator import search_all
from research_assistant.sources.domains import DOMAIN_SOURCES, resolve_sources
from research_assistant.synthesis import review as synthesis_review

# Valid --domain choices: all preset keys plus "auto".
_DOMAIN_CHOICES = list(DOMAIN_SOURCES) + ["auto"]


def _cmd_search(args: argparse.Namespace) -> int:
    source_names, explanation = resolve_sources(args.domain, args.sources, args.query)
    print(f"[{explanation}]")

    papers, warnings = search_all(args.query, limit=args.limit, source_names=source_names)

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if not papers:
        print("No results.")
        return 1

    # Remember these results so `research save <number>` can refer to them.
    session.save_last_search(papers)

    for i, paper in enumerate(papers, start=1):
        print(f"{i}. {paper.display()}\n")

    print(f"({len(papers)} results -- use `research save <number>` to save one)")
    return 0


def _cmd_save(args: argparse.Namespace) -> int:
    paper = session.resolve_paper_ref(args.ref)
    if paper is None:
        print(
            f"Could not resolve {args.ref!r}. Run `research search` first and "
            "use one of its result numbers, or pass a full paper id "
            "(e.g. arxiv:2301.12345).",
            file=sys.stderr,
        )
        return 1

    print(f"Saving: {paper.title}")
    print(
        "(loading the local embedding model -- first run downloads it, "
        "this can take a minute)"
    )
    chunks = ingest.save_paper(paper, full_text=args.full_text)
    print(f"Saved {paper.id} as {chunks} chunk(s).")
    return 0


def _cmd_library(args: argparse.Namespace) -> int:
    papers = store.list_saved_papers()
    if not papers:
        print("Your library is empty. Use `research search` then `research save <n>`.")
        return 0

    for i, meta in enumerate(sorted(papers, key=lambda p: p.get("title", "")), start=1):
        year = f" ({meta['year']})" if meta.get("year") else ""
        print(f"{i}. [{meta.get('source')}] {meta.get('title')}{year}")
        if meta.get("authors"):
            print(f"   {meta['authors']}")
        if meta.get("url"):
            print(f"   {meta['url']}")

    print(f"\n{len(papers)} paper(s), {store.chunk_count()} chunk(s) total.")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    answer = qa_engine.ask(
        args.question,
        top_k=args.top_k,
        use_hybrid=not args.no_hybrid,
        use_reranker=not args.no_reranker,
    )
    print(answer.text)
    if answer.citations:
        print("\nSources:")
        for citation in answer.citations:
            year = f" ({citation.year})" if citation.year else ""
            print(f"  - {citation.title}{year} -- {citation.authors}")
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    if args.no_live:
        print(f'Gathering papers on "{args.topic}" [library only, no live search]...')
        lit = synthesis_review.review(
            args.topic,
            live_limit=args.live_limit,
            include_live=False,
            source_names=None,
        )
    else:
        source_names, explanation = resolve_sources(args.domain, args.sources, args.topic)
        print(f'Gathering papers on "{args.topic}" [{explanation}]...')
        lit = synthesis_review.review(
            args.topic,
            live_limit=args.live_limit,
            include_live=True,
            source_names=source_names,
        )

    if not lit.sources:
        print(lit.text)
        return 1

    print(f"Synthesising across {lit.paper_count} paper(s)...\n")
    print(f"=== Literature Review: {args.topic!r} ===\n")
    print(lit.text)
    print(f"\n=== Sources ({lit.paper_count} papers) ===")
    for i, src in enumerate(lit.sources, 1):
        print(f"  {i}. [{src.source}] {src.title} — {src.label}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research",
        description="An AI-powered research assistant over scholarly papers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser(
        "search", help="Search live paper sources (arXiv, Semantic Scholar, OpenAlex)"
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Max results per source (default: 5)"
    )
    search_parser.add_argument(
        "--domain",
        choices=_DOMAIN_CHOICES,
        default="auto",
        help=(
            "Topic domain for source routing (default: auto). "
            "'auto' classifies the query locally with the embedding model. "
            "'all' queries every source. "
            f"Presets: {', '.join(k for k in DOMAIN_SOURCES if k != 'all')}"
        ),
    )
    search_parser.add_argument(
        "--sources",
        default=None,
        help=(
            "Comma-separated sources to search, overrides --domain. "
            "Choices: arxiv, semantic_scholar, openalex"
        ),
    )
    search_parser.set_defaults(func=_cmd_search)

    save_parser = subparsers.add_parser(
        "save", help="Save a search result to your personal library"
    )
    save_parser.add_argument(
        "ref", help="Result number from the last search, or a full paper id"
    )
    save_parser.add_argument(
        "--full-text",
        action="store_true",
        help="Also extract and embed the full PDF text, not just title + abstract "
        "(richer Q&A, but can take several minutes per paper on CPU)",
    )
    save_parser.set_defaults(func=_cmd_save)

    library_parser = subparsers.add_parser(
        "library", help="List papers saved in your personal library"
    )
    library_parser.set_defaults(func=_cmd_library)

    ask_parser = subparsers.add_parser(
        "ask", help="Ask a cited question over your saved papers"
    )
    ask_parser.add_argument("question", help="Your question")
    ask_parser.add_argument(
        "--top-k", type=int, default=6, help="Final chunks forwarded to Claude (default: 6)"
    )
    ask_parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Use vector-only retrieval instead of BM25 + vector fusion (A/B comparison)",
    )
    ask_parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip cross-encoder reranking, use raw retrieval scores (A/B comparison)",
    )
    ask_parser.set_defaults(func=_cmd_ask)

    review_parser = subparsers.add_parser(
        "review", help="Synthesise a literature review across live search + saved library"
    )
    review_parser.add_argument("topic", help="Research topic to review")
    review_parser.add_argument(
        "--live-limit",
        type=int,
        default=10,
        help="Papers to fetch per live source (arXiv, S2, OpenAlex). Default: 10",
    )
    review_parser.add_argument(
        "--no-live",
        action="store_true",
        help="Skip live API search; synthesise only from your saved library",
    )
    review_parser.add_argument(
        "--domain",
        choices=_DOMAIN_CHOICES,
        default="auto",
        help=(
            "Topic domain for source routing (default: auto). "
            "'auto' classifies the topic locally with the embedding model. "
            "'all' queries every source. "
            f"Presets: {', '.join(k for k in DOMAIN_SOURCES if k != 'all')}"
        ),
    )
    review_parser.add_argument(
        "--sources",
        default=None,
        help=(
            "Comma-separated sources to query, overrides --domain. "
            "Choices: arxiv, semantic_scholar, openalex"
        ),
    )
    review_parser.set_defaults(func=_cmd_review)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles often default to a legacy code page that can't render
    # non-ASCII author names/abstracts (e.g. "Kröker" -> "Kr?ker"). Force
    # UTF-8 if the stream supports reconfiguring.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
