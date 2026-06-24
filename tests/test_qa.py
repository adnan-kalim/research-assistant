"""Tests for pure-logic helpers in the Q&A engine (no network, no model)."""

from __future__ import annotations

from research_assistant.qa.engine import _label


def test_label_single_author():
    assert _label({"authors": "Alice Smith", "year": 2023}) == "Alice Smith, 2023"


def test_label_multiple_authors_uses_et_al():
    label = _label({"authors": "Alice Smith, Bob Jones", "year": 2023})
    assert label == "Alice Smith et al., 2023"


def test_label_missing_authors_and_year():
    assert _label({}) == "Unknown, n.d."
