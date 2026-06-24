"""OpenAlex source -- no API key required, fully open index of ~250M works.

API docs: https://docs.openalex.org/api-entities/works
"""

from __future__ import annotations

import requests

from research_assistant.config import settings
from research_assistant.sources.base import Paper, PaperSource

_SEARCH_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]} to save space.

    Flip that back into normal, readable text.
    """
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions.append((idx, word))
    positions.sort()
    return " ".join(word for _, word in positions)


class OpenAlexSource(PaperSource):
    name = "openalex"

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        params: dict[str, str | int] = {"search": query, "per_page": limit}
        if settings.contact_email:
            # Joining OpenAlex's "polite pool" gets faster, less-throttled
            # responses -- entirely optional.
            params["mailto"] = settings.contact_email

        response = requests.get(
            _SEARCH_URL,
            params=params,
            headers={"User-Agent": "research-assistant/0.1"},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        papers: list[Paper] = []
        for item in data.get("results", []):
            authors = [
                (authorship.get("author") or {}).get("display_name", "")
                for authorship in item.get("authorships") or []
            ]
            primary_location = item.get("primary_location") or {}
            open_access = item.get("open_access") or {}
            work_id = (item.get("id") or "").rsplit("/", 1)[-1]

            papers.append(
                Paper(
                    id=f"openalex:{work_id}",
                    source=self.name,
                    title=item.get("title") or item.get("display_name") or "(untitled)",
                    abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
                    authors=[name for name in authors if name],
                    year=item.get("publication_year"),
                    url=primary_location.get("landing_page_url") or item.get("id"),
                    pdf_url=open_access.get("oa_url"),
                )
            )
        return papers
