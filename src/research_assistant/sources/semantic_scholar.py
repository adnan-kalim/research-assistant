"""Semantic Scholar source -- works without a key (lower rate limit) or with
an optional S2_API_KEY (set in .env) for higher limits.

API docs: https://api.semanticscholar.org/api-docs/graph
"""

from __future__ import annotations

import requests

from research_assistant.config import settings
from research_assistant.sources.base import Paper, PaperSource

_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,abstract,authors,year,url,externalIds,openAccessPdf"


class SemanticScholarSource(PaperSource):
    name = "semantic_scholar"

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        headers = {}
        if settings.semantic_scholar_api_key:
            headers["x-api-key"] = settings.semantic_scholar_api_key

        response = requests.get(
            _SEARCH_URL,
            params={"query": query, "limit": limit, "fields": _FIELDS},
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        papers: list[Paper] = []
        for item in data.get("data", []):
            pdf = (item.get("openAccessPdf") or {}).get("url")
            papers.append(
                Paper(
                    id=f"s2:{item.get('paperId')}",
                    source=self.name,
                    title=item.get("title") or "(untitled)",
                    abstract=item.get("abstract"),
                    authors=[a.get("name", "") for a in item.get("authors") or []],
                    year=item.get("year"),
                    url=item.get("url"),
                    pdf_url=pdf,
                )
            )
        return papers
