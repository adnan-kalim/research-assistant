"""Tests for ingest logic (chunking + metadata building), with the slow
embedding model and Chroma store mocked out -- this tests *our* logic,
not the third-party libraries underneath it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from research_assistant.library import ingest
from research_assistant.sources.base import Paper


def test_save_paper_builds_expected_metadata_and_dedupes(monkeypatch):
    fake_index = MagicMock()
    deleted_ids: list[str] = []

    monkeypatch.setattr(ingest.store, "get_index", lambda: fake_index)
    monkeypatch.setattr(ingest.store, "delete_paper", lambda pid: deleted_ids.append(pid))
    monkeypatch.setattr(ingest, "_try_fetch_full_text", lambda paper: None)

    paper = Paper(
        id="arxiv:1234.5678",
        source="arxiv",
        title="A Great Paper",
        abstract="This paper is about something important.",
        authors=["Alice Smith", "Bob Jones"],
        year=2024,
        url="http://arxiv.org/abs/1234.5678",
    )

    chunks_written = ingest.save_paper(paper)

    # De-dup must happen before inserting new chunks.
    assert deleted_ids == ["arxiv:1234.5678"]
    assert chunks_written > 0

    fake_index.insert_nodes.assert_called_once()
    nodes = fake_index.insert_nodes.call_args.args[0]
    assert len(nodes) == chunks_written

    meta = nodes[0].metadata
    assert meta["paper_id"] == "arxiv:1234.5678"
    assert meta["title"] == "A Great Paper"
    assert meta["authors"] == "Alice Smith, Bob Jones"
    assert meta["year"] == 2024
