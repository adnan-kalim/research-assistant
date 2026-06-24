"""Fan a search out across all live sources, merging results.

This is what makes the "hybrid corpus" idea concrete: instead of hosting
millions of papers ourselves, we query the indexes that already have them.
If one source is down or rate-limited, the others still return results --
a single flaky network call should never fail the whole search.
"""

from __future__ import annotations

from research_assistant.sources.arxiv_client import ArxivSource
from research_assistant.sources.base import Paper, PaperSource
from research_assistant.sources.openalex import OpenAlexSource
from research_assistant.sources.semantic_scholar import SemanticScholarSource

ALL_SOURCES: dict[str, PaperSource] = {
    "arxiv": ArxivSource(),
    "semantic_scholar": SemanticScholarSource(),
    "openalex": OpenAlexSource(),
}


def search_all(
    query: str,
    limit: int = 10,
    source_names: list[str] | None = None,
) -> tuple[list[Paper], list[str]]:
    """Search the requested sources (default: all) and merge results.

    Returns `(papers, warnings)`. A source that raises (network error, rate
    limit, bad response, ...) is skipped and noted in `warnings` rather than
    aborting the whole search.
    """
    names = source_names or list(ALL_SOURCES)
    papers: list[Paper] = []
    warnings: list[str] = []

    for name in names:
        source = ALL_SOURCES.get(name)
        if source is None:
            warnings.append(f"Unknown source: {name!r}")
            continue
        try:
            papers.extend(source.search(query, limit=limit))
        except Exception as exc:  # noqa: BLE001 -- deliberate: one flaky
            # third-party API must never take down the whole search.
            warnings.append(f"{name} failed: {exc}")

    return papers, warnings
