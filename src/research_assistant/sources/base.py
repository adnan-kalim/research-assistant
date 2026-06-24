"""Shared data model and interface for live scholarly-paper sources.

Every source (arXiv, Semantic Scholar, OpenAlex, ...) speaks a different API
with different field names. Each source module is responsible for
normalizing its results into the `Paper` shape below, so the rest of the
codebase (CLI, library, Q&A) never has to know which API a paper came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


@dataclass
class Paper:
    """A paper, normalized to a common shape regardless of its source."""

    id: str
    """Stable identifier, e.g. "arxiv:2301.12345" or "s2:abcd1234"."""

    source: str
    """Which API this came from: "arxiv" | "semantic_scholar" | "openalex"."""

    title: str
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    url: str | None = None
    """Human-readable landing page for the paper."""

    pdf_url: str | None = None
    """Direct PDF link, if one is freely available."""

    def short_authors(self, max_authors: int = 3) -> str:
        if not self.authors:
            return "Unknown authors"
        if len(self.authors) <= max_authors:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:max_authors]) + " et al."

    def display(self) -> str:
        """A short, human-readable summary block for printing in the CLI."""
        year = f" ({self.year})" if self.year else ""
        lines = [f"[{self.source}] {self.title}{year}", f"  {self.short_authors()}"]
        if self.abstract:
            snippet = self.abstract.strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:277] + "..."
            lines.append(f"  {snippet}")
        if self.url:
            lines.append(f"  {self.url}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Paper":
        return cls(**data)


class PaperSource(ABC):
    """Common interface every live search source implements."""

    name: str

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[Paper]:
        """Return up to `limit` papers matching `query`, best matches first."""
        raise NotImplementedError
