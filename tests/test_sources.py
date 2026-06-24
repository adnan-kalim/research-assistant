"""Tests for the Paper dataclass and the multi-source aggregator."""

from __future__ import annotations

import research_assistant.sources.aggregator as aggregator
from research_assistant.sources.aggregator import search_all
from research_assistant.sources.base import Paper, PaperSource


def test_paper_round_trips_through_dict():
    paper = Paper(
        id="arxiv:1234.5678",
        source="arxiv",
        title="Test Paper",
        abstract="An abstract.",
        authors=["A. Author"],
        year=2024,
        url="http://example.com",
    )
    restored = Paper.from_dict(paper.to_dict())
    assert restored == paper


def test_paper_display_truncates_long_abstracts():
    paper = Paper(id="x", source="test", title="T", abstract="a" * 500)
    text = paper.display()
    assert "..." in text


class _FlakySource(PaperSource):
    name = "flaky"

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        raise RuntimeError("simulated network failure")


class _WorkingSource(PaperSource):
    name = "working"

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        return [Paper(id="working:1", source=self.name, title="Found it")]


def test_search_all_degrades_gracefully_when_one_source_fails(monkeypatch):
    monkeypatch.setitem(aggregator.ALL_SOURCES, "flaky", _FlakySource())
    monkeypatch.setitem(aggregator.ALL_SOURCES, "working", _WorkingSource())

    papers, warnings = search_all("anything", source_names=["flaky", "working"])

    assert len(papers) == 1
    assert papers[0].id == "working:1"
    assert any("flaky failed" in w for w in warnings)
