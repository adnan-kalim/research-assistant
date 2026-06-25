"""Tests for Phase 5 synthesis / literature review.

All Claude API calls and store interactions are mocked so tests run fast with
no network or GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from research_assistant.synthesis.review import (
    LitReview,
    PaperSummary,
    _RawPaper,
    _gather_papers,
    _make_label,
    _map_batch,
    _reduce,
    review,
)


# ---------------------------------------------------------------------------
# _make_label
# ---------------------------------------------------------------------------


def test_make_label_single_author():
    assert _make_label("Alice Smith", 2023) == "Alice Smith, 2023"


def test_make_label_multiple_authors():
    assert _make_label("Alice Smith, Bob Jones", 2021) == "Alice Smith et al., 2021"


def test_make_label_no_authors():
    assert _make_label("", None) == "Unknown, n.d."


def test_make_label_no_year():
    assert _make_label("Alice Smith", None) == "Alice Smith, n.d."


# ---------------------------------------------------------------------------
# _gather_papers: deduplication and source merging
# ---------------------------------------------------------------------------


def _make_paper(paper_id, title="T", abstract="A", authors=None, year=2024, source="arxiv"):
    from research_assistant.sources.base import Paper
    return Paper(
        id=paper_id,
        source=source,
        title=title,
        abstract=abstract,
        authors=authors or ["Author One"],
        year=year,
    )


def test_gather_papers_live_only(monkeypatch):
    """Live papers are returned when the library is empty."""
    from research_assistant.synthesis import review as rev_module

    live_papers = [_make_paper("arxiv:001", "Paper A"), _make_paper("arxiv:002", "Paper B")]
    monkeypatch.setattr(rev_module, "search_all", lambda q, limit, source_names=None: (live_papers, []))
    monkeypatch.setattr(rev_module.store, "get_paper_texts", lambda: {})
    monkeypatch.setattr(rev_module.store, "list_saved_papers", lambda: [])

    result = _gather_papers("topic", live_limit=10, include_live=True)

    assert len(result) == 2
    assert {r.paper_id for r in result} == {"arxiv:001", "arxiv:002"}


def test_gather_papers_library_only(monkeypatch):
    """With include_live=False, only library papers are returned."""
    from research_assistant.synthesis import review as rev_module

    saved_meta = [{"paper_id": "lib:001", "title": "Saved Paper", "authors": "A", "year": 2022, "source": "arxiv"}]
    monkeypatch.setattr(rev_module.store, "get_paper_texts", lambda: {"lib:001": "Full text here."})
    monkeypatch.setattr(rev_module.store, "list_saved_papers", lambda: saved_meta)

    result = _gather_papers("topic", live_limit=10, include_live=False)

    assert len(result) == 1
    assert result[0].paper_id == "lib:001"
    assert result[0].text == "Full text here."


def test_gather_papers_deduplication_live_wins(monkeypatch):
    """A paper in both live results and library appears only once (live version wins)."""
    from research_assistant.synthesis import review as rev_module

    live_papers = [_make_paper("arxiv:001", "Live Title", abstract="Live abstract")]
    saved_meta = [{"paper_id": "arxiv:001", "title": "Saved Title", "authors": "A", "year": 2024, "source": "arxiv"}]

    monkeypatch.setattr(rev_module, "search_all", lambda q, limit, source_names=None: (live_papers, []))
    monkeypatch.setattr(rev_module.store, "get_paper_texts", lambda: {"arxiv:001": "Saved text"})
    monkeypatch.setattr(rev_module.store, "list_saved_papers", lambda: saved_meta)

    result = _gather_papers("topic", live_limit=10, include_live=True)

    assert len(result) == 1
    assert result[0].title == "Live Title"   # live version kept


def test_gather_papers_passes_source_names_to_search_all(monkeypatch):
    """source_names should be forwarded to search_all so domain routing takes effect."""
    from research_assistant.synthesis import review as rev_module

    received_source_names: list = []

    def fake_search_all(q, limit, source_names=None):
        received_source_names.append(source_names)
        return ([], [])

    monkeypatch.setattr(rev_module, "search_all", fake_search_all)
    monkeypatch.setattr(rev_module.store, "get_paper_texts", lambda: {})
    monkeypatch.setattr(rev_module.store, "list_saved_papers", lambda: [])

    _gather_papers("topic", live_limit=5, include_live=True, source_names=["semantic_scholar"])

    assert received_source_names == [["semantic_scholar"]]


# ---------------------------------------------------------------------------
# _map_batch: response parsing
# ---------------------------------------------------------------------------


def _mock_client_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_map_batch_parses_numbered_blocks():
    """MAP response with [1]/[2] markers should be split into per-paper summaries."""
    client = MagicMock()
    client.messages.create.return_value = _mock_client_response(
        "[1] Paper one focuses on attention.\n\n[2] Paper two addresses efficiency."
    )

    papers = [
        _RawPaper("p1", "Title 1", "A", 2023, "arxiv", "Abstract 1"),
        _RawPaper("p2", "Title 2", "B", 2022, "arxiv", "Abstract 2"),
    ]
    summaries = _map_batch(papers, "topic", client)

    assert summaries[0] == "Paper one focuses on attention."
    assert summaries[1] == "Paper two addresses efficiency."


def test_map_batch_handles_missing_block():
    """If Claude omits a paper number, that slot gets an empty string."""
    client = MagicMock()
    client.messages.create.return_value = _mock_client_response("[1] Only first paper.")

    papers = [
        _RawPaper("p1", "T1", "A", 2023, "arxiv", "text"),
        _RawPaper("p2", "T2", "B", 2023, "arxiv", "text"),
    ]
    summaries = _map_batch(papers, "topic", client)

    assert summaries[0] == "Only first paper."
    assert summaries[1] == ""


def test_map_batch_truncates_long_text():
    """Abstracts longer than _MAX_TEXT_CHARS should be truncated in the prompt."""
    from research_assistant.synthesis.review import _MAX_TEXT_CHARS

    client = MagicMock()
    client.messages.create.return_value = _mock_client_response("[1] Summary.")

    long_text = "x" * (_MAX_TEXT_CHARS + 500)
    papers = [_RawPaper("p1", "T", "A", 2023, "arxiv", long_text)]
    _map_batch(papers, "topic", client)

    call_args = client.messages.create.call_args
    user_content = call_args.kwargs["messages"][0]["content"]
    assert "x" * (_MAX_TEXT_CHARS + 1) not in user_content


# ---------------------------------------------------------------------------
# _reduce: synthesis call
# ---------------------------------------------------------------------------


def test_reduce_passes_all_summaries_to_claude():
    """REDUCE should include every paper summary in the user message."""
    client = MagicMock()
    client.messages.create.return_value = _mock_client_response("Synthesis result.")

    summaries = [
        PaperSummary("p1", "Smith, 2023", "Title A", "arxiv", "Summary A"),
        PaperSummary("p2", "Jones, 2022", "Title B", "arxiv", "Summary B"),
    ]
    text = _reduce(summaries, "neural networks", client)

    assert text == "Synthesis result."
    call_kwargs = client.messages.create.call_args.kwargs
    user_msg = call_kwargs["messages"][0]["content"]
    assert "Smith, 2023" in user_msg
    assert "Jones, 2022" in user_msg
    assert "neural networks" in user_msg


# ---------------------------------------------------------------------------
# review(): high-level flow
# ---------------------------------------------------------------------------


def test_review_returns_early_no_api_key(monkeypatch):
    from research_assistant.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", None)
    result = review("topic")
    assert "ANTHROPIC_API_KEY" in result.text
    assert result.paper_count == 0


def test_review_returns_early_no_papers(monkeypatch):
    from research_assistant.synthesis import review as rev_module
    from research_assistant.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "sk-fake")
    monkeypatch.setattr(rev_module, "_rewrite", lambda t: [t])   # skip model load
    monkeypatch.setattr(rev_module, "search_all", lambda q, limit, source_names=None: ([], []))
    monkeypatch.setattr(rev_module.store, "get_paper_texts", lambda: {})
    monkeypatch.setattr(rev_module.store, "list_saved_papers", lambda: [])

    result = review("obscure topic nobody wrote about")
    assert result.paper_count == 0
    assert "No papers found" in result.text


def test_review_full_map_reduce_flow(monkeypatch):
    """End-to-end: gather → MAP → REDUCE produces a LitReview with sources."""
    from research_assistant.synthesis import review as rev_module
    from research_assistant.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "sk-fake")
    monkeypatch.setattr(rev_module, "_rewrite", lambda t: [t])   # skip model load

    live = [_make_paper("arxiv:001", "Paper Alpha"), _make_paper("arxiv:002", "Paper Beta")]
    monkeypatch.setattr(rev_module, "search_all", lambda q, limit, source_names=None: (live, []))
    monkeypatch.setattr(rev_module.store, "get_paper_texts", lambda: {})
    monkeypatch.setattr(rev_module.store, "list_saved_papers", lambda: [])

    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:   # MAP call
            return _mock_client_response("[1] Alpha summary.\n\n[2] Beta summary.")
        else:                       # REDUCE call
            return _mock_client_response("## Main Themes\nBoth papers cover X.")

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = fake_create
    monkeypatch.setattr(rev_module.anthropic, "Anthropic", lambda **kw: mock_client)

    result = review("transformers", live_limit=5)

    assert result.paper_count == 2
    assert "## Main Themes" in result.text
    assert result.sources[0].summary == "Alpha summary."
    assert result.sources[1].summary == "Beta summary."
    assert call_count["n"] == 2   # exactly one MAP + one REDUCE call
