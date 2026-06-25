"""Tests for domain-aware source routing (sources/domains.py).

All embedding model calls are mocked so tests run fast with no GPU or network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from research_assistant.sources.domains import (
    DOMAIN_DESCRIPTIONS,
    DOMAIN_SOURCES,
    _cosine_similarity,
    classify_domain,
    resolve_sources,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(size: int, hot: int) -> list[float]:
    """Return a unit vector with 1.0 at position *hot*, 0.0 elsewhere."""
    v = [0.0] * size
    v[hot] = 1.0
    return v


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal_vectors():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert abs(_cosine_similarity(a, b)) < 1e-6


def test_cosine_zero_vector_returns_zero():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# classify_domain: nearest neighbour in embedding space
# ---------------------------------------------------------------------------


def test_classify_domain_returns_nearest(monkeypatch):
    """The domain whose description vector is closest should be returned."""
    from research_assistant.sources import domains as dom_module

    domains = list(DOMAIN_DESCRIPTIONS.keys())
    n = len(domains)

    # Assign each domain its own orthogonal unit vector.
    domain_vecs = {d: _unit_vec(n, i) for i, d in enumerate(domains)}

    # Topic embedding points exactly at "psychology" (index 6 in alphabetical order).
    target_domain = "psychology"
    target_idx = domains.index(target_domain)
    topic_vec = _unit_vec(n, target_idx)

    mock_embed = MagicMock()
    mock_embed.get_text_embedding.return_value = topic_vec

    # Patch cached vectors and embed model.
    monkeypatch.setattr(dom_module, "_domain_vectors", domain_vecs)
    monkeypatch.setattr(dom_module.store, "get_embed_model", lambda: mock_embed)

    result = classify_domain("attention spans among teenagers")
    assert result == target_domain


def test_classify_domain_cs_topic(monkeypatch):
    """A CS-ish topic should be routed to 'cs' when its vector is closest."""
    from research_assistant.sources import domains as dom_module

    domains = list(DOMAIN_DESCRIPTIONS.keys())
    n = len(domains)
    domain_vecs = {d: _unit_vec(n, i) for i, d in enumerate(domains)}

    cs_idx = domains.index("cs")
    topic_vec = _unit_vec(n, cs_idx)

    mock_embed = MagicMock()
    mock_embed.get_text_embedding.return_value = topic_vec
    monkeypatch.setattr(dom_module, "_domain_vectors", domain_vecs)
    monkeypatch.setattr(dom_module.store, "get_embed_model", lambda: mock_embed)

    result = classify_domain("diffusion models for image generation")
    assert result == "cs"


# ---------------------------------------------------------------------------
# resolve_sources: priority chain
# ---------------------------------------------------------------------------


def test_resolve_explicit_sources_overrides_domain(monkeypatch):
    """Explicit --sources takes priority over --domain."""
    names, explanation = resolve_sources(
        domain="cs",
        sources="openalex",
        topic="anything",
    )
    assert names == ["openalex"]
    assert "explicit" in explanation


def test_resolve_explicit_sources_multiple(monkeypatch):
    names, explanation = resolve_sources(
        domain=None,
        sources="arxiv,openalex",
        topic="anything",
    )
    assert set(names) == {"arxiv", "openalex"}
    assert "explicit" in explanation


def test_resolve_explicit_sources_unknown_filtered(monkeypatch):
    """Unknown source names in --sources should be silently filtered out."""
    names, explanation = resolve_sources(
        domain=None,
        sources="arxiv,nonexistent_source",
        topic="anything",
    )
    assert "arxiv" in names
    assert "nonexistent_source" not in names
    assert "unknown" in explanation


def test_resolve_explicit_sources_all_unknown_falls_back():
    """If every --sources name is unknown, fall back to all sources."""
    names, _ = resolve_sources(domain=None, sources="made_up", topic="X")
    from research_assistant.sources.aggregator import ALL_SOURCES
    assert set(names) == set(ALL_SOURCES)


def test_resolve_preset_domain_psychology():
    names, explanation = resolve_sources(domain="psychology", sources=None, topic="X")
    assert set(names) == {"semantic_scholar", "openalex"}
    assert "arxiv" not in names
    assert "psychology" in explanation


def test_resolve_preset_domain_cs():
    names, explanation = resolve_sources(domain="cs", sources=None, topic="X")
    assert "arxiv" in names
    assert "semantic_scholar" in names
    assert "cs" in explanation


def test_resolve_preset_domain_all():
    names, explanation = resolve_sources(domain="all", sources=None, topic="X")
    from research_assistant.sources.aggregator import ALL_SOURCES
    assert set(names) == set(ALL_SOURCES)


def test_resolve_unknown_domain_falls_back_to_all():
    names, explanation = resolve_sources(domain="underwater_basket_weaving", sources=None, topic="X")
    from research_assistant.sources.aggregator import ALL_SOURCES
    assert set(names) == set(ALL_SOURCES)
    assert "unknown" in explanation


def test_resolve_auto_calls_classify_domain(monkeypatch):
    """With domain='auto', resolve_sources should call classify_domain."""
    from research_assistant.sources import domains as dom_module

    called_with: list[str] = []

    def fake_classify(topic):
        called_with.append(topic)
        return "medicine"

    monkeypatch.setattr(dom_module, "classify_domain", fake_classify)

    names, explanation = resolve_sources(domain="auto", sources=None, topic="cancer drug trials")

    assert called_with == ["cancer drug trials"]
    assert set(names) == set(DOMAIN_SOURCES["medicine"])
    assert "auto-detected" in explanation
    assert "medicine" in explanation


def test_resolve_none_domain_also_auto_detects(monkeypatch):
    """domain=None should behave the same as domain='auto'."""
    from research_assistant.sources import domains as dom_module

    monkeypatch.setattr(dom_module, "classify_domain", lambda t: "biology")

    names, explanation = resolve_sources(domain=None, sources=None, topic="gene expression")
    assert set(names) == set(DOMAIN_SOURCES["biology"])
    assert "auto-detected" in explanation


def test_resolve_auto_excludes_arxiv_for_social_science(monkeypatch):
    """Auto-detected social_science must exclude arXiv from the source list."""
    from research_assistant.sources import domains as dom_module

    monkeypatch.setattr(dom_module, "classify_domain", lambda t: "social_science")

    names, explanation = resolve_sources(domain="auto", sources=None, topic="inequality and poverty")
    assert "arxiv" not in names
    assert "arxiv" in explanation   # mentioned as excluded


# ---------------------------------------------------------------------------
# DOMAIN_SOURCES / DOMAIN_DESCRIPTIONS sanity checks
# ---------------------------------------------------------------------------


def test_all_domain_sources_entries_are_valid():
    from research_assistant.sources.aggregator import ALL_SOURCES
    for domain, sources in DOMAIN_SOURCES.items():
        for s in sources:
            assert s in ALL_SOURCES, f"{domain}: unknown source {s!r}"


def test_domain_descriptions_covers_all_non_all_domains():
    expected = {k for k in DOMAIN_SOURCES if k != "all"}
    assert set(DOMAIN_DESCRIPTIONS.keys()) == expected
