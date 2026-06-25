"""Tests for query_rewriter.rewrite().

All model calls are mocked — tests run with no GPU, no network, no downloads.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_pipe_output(text: str):
    """Build the list-of-dicts structure that transformers pipeline returns."""
    return [{"generated_text": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": text},
    ]}]


# ---------------------------------------------------------------------------
# rewrite(): output parsing
# ---------------------------------------------------------------------------


def test_rewrite_returns_three_clean_queries(monkeypatch):
    from research_assistant.sources import query_rewriter as qr

    mock_pipe = MagicMock(return_value=_make_pipe_output(
        "adolescent attention span\nteenage focus digital media\nyouth sustained attention"
    ))
    monkeypatch.setattr(qr, "_pipeline", mock_pipe)

    result = qr.rewrite("attention spans among teenagers in the last ten years")

    assert result == [
        "adolescent attention span",
        "teenage focus digital media",
        "youth sustained attention",
    ]


def test_rewrite_strips_leading_numbers(monkeypatch):
    from research_assistant.sources import query_rewriter as qr

    mock_pipe = MagicMock(return_value=_make_pipe_output(
        "1. adolescent attention span\n2. teenage digital media\n3. youth focus"
    ))
    monkeypatch.setattr(qr, "_pipeline", mock_pipe)

    result = qr.rewrite("topic")

    assert result[0] == "adolescent attention span"
    assert not result[0].startswith("1")


def test_rewrite_strips_bullet_characters(monkeypatch):
    from research_assistant.sources import query_rewriter as qr

    mock_pipe = MagicMock(return_value=_make_pipe_output(
        "- query one\n• query two\n* query three"
    ))
    monkeypatch.setattr(qr, "_pipeline", mock_pipe)

    result = qr.rewrite("topic")

    assert result == ["query one", "query two", "query three"]


def test_rewrite_strips_quotes(monkeypatch):
    from research_assistant.sources import query_rewriter as qr

    mock_pipe = MagicMock(return_value=_make_pipe_output(
        '"query one"\n"query two"\n"query three"'
    ))
    monkeypatch.setattr(qr, "_pipeline", mock_pipe)

    result = qr.rewrite("topic")

    assert result == ["query one", "query two", "query three"]


def test_rewrite_caps_at_three_even_if_model_gives_more(monkeypatch):
    from research_assistant.sources import query_rewriter as qr

    mock_pipe = MagicMock(return_value=_make_pipe_output(
        "q1\nq2\nq3\nq4\nq5"
    ))
    monkeypatch.setattr(qr, "_pipeline", mock_pipe)

    result = qr.rewrite("topic")

    assert len(result) == 3


def test_rewrite_skips_empty_lines(monkeypatch):
    from research_assistant.sources import query_rewriter as qr

    mock_pipe = MagicMock(return_value=_make_pipe_output(
        "query one\n\n\nquery two\n\nquery three"
    ))
    monkeypatch.setattr(qr, "_pipeline", mock_pipe)

    result = qr.rewrite("topic")

    assert "" not in result
    assert len(result) == 3


# ---------------------------------------------------------------------------
# rewrite(): error handling
# ---------------------------------------------------------------------------


def test_rewrite_falls_back_to_original_topic_on_exception(monkeypatch):
    from research_assistant.sources import query_rewriter as qr

    def exploding_pipe(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(qr, "_pipeline", exploding_pipe)

    result = qr.rewrite("my topic")

    assert result == ["my topic"]


def test_rewrite_falls_back_when_model_returns_empty(monkeypatch):
    from research_assistant.sources import query_rewriter as qr

    mock_pipe = MagicMock(return_value=_make_pipe_output("   \n  \n  "))
    monkeypatch.setattr(qr, "_pipeline", mock_pipe)

    result = qr.rewrite("my topic")

    assert result == ["my topic"]
