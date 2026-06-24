"""arXiv source -- no API key required.

Uses the official `arxiv` Python package, which wraps arXiv's public Atom
feed API.
"""

from __future__ import annotations

import arxiv

from research_assistant.sources.base import Paper, PaperSource


class ArxivSource(PaperSource):
    name = "arxiv"

    def __init__(self) -> None:
        self._client = arxiv.Client()

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: list[Paper] = []
        for result in self._client.results(search):
            # arxiv IDs look like "2301.12345v2" -- strip the version suffix
            # so re-fetching the same paper later always yields the same id.
            short_id = result.get_short_id().split("v")[0]
            papers.append(
                Paper(
                    id=f"arxiv:{short_id}",
                    source=self.name,
                    title=result.title.strip(),
                    abstract=result.summary.strip() if result.summary else None,
                    authors=[author.name for author in result.authors],
                    year=result.published.year if result.published else None,
                    url=result.entry_id,
                    pdf_url=result.pdf_url,
                )
            )
        return papers
