"""Tests for Phase 4 retrieval quality improvements.

All tests mock heavy dependencies (Chroma, models, BM25/reranker) so they
run fast with no GPU or network access. We test *our* logic -- wiring,
branching, and edge cases -- not the third-party retrieval libraries.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# store.get_all_text_nodes
# ---------------------------------------------------------------------------


def test_get_all_text_nodes_returns_populated_nodes():
    from research_assistant.library import store

    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "documents": ["Text of chunk one.", "Text of chunk two."],
        "metadatas": [{"paper_id": "p1", "title": "A"}, {"paper_id": "p2", "title": "B"}],
        "ids": ["chunk-1", "chunk-2"],
    }

    with patch.object(store, "_get_collection", return_value=mock_collection):
        nodes = store.get_all_text_nodes()

    assert len(nodes) == 2
    assert nodes[0].text == "Text of chunk one."
    assert nodes[0].metadata["paper_id"] == "p1"
    assert nodes[0].id_ == "chunk-1"
    assert nodes[1].text == "Text of chunk two."


def test_get_all_text_nodes_skips_falsy_documents():
    from research_assistant.library import store

    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "documents": ["Real text.", "", None],
        "metadatas": [{"paper_id": "p1"}, {"paper_id": "p2"}, {"paper_id": "p3"}],
        "ids": ["c1", "c2", "c3"],
    }

    with patch.object(store, "_get_collection", return_value=mock_collection):
        nodes = store.get_all_text_nodes()

    assert len(nodes) == 1
    assert nodes[0].id_ == "c1"


def test_get_all_text_nodes_empty_library():
    from research_assistant.library import store

    mock_collection = MagicMock()
    mock_collection.get.return_value = {"documents": [], "metadatas": [], "ids": []}

    with patch.object(store, "_get_collection", return_value=mock_collection):
        nodes = store.get_all_text_nodes()

    assert nodes == []


# ---------------------------------------------------------------------------
# _label helper (unchanged from Phase 3, regression guard)
# ---------------------------------------------------------------------------


def test_label_single_author():
    from research_assistant.qa.engine import _label

    assert _label({"authors": "Alice Smith", "year": 2023}) == "Alice Smith, 2023"


def test_label_multiple_authors_uses_et_al():
    from research_assistant.qa.engine import _label

    assert _label({"authors": "Alice Smith, Bob Jones", "year": 2023}) == "Alice Smith et al., 2023"


def test_label_missing_fields():
    from research_assistant.qa.engine import _label

    assert _label({}) == "Unknown, n.d."


# ---------------------------------------------------------------------------
# _retrieve_nodes: branching behaviour
# ---------------------------------------------------------------------------


def _make_mock_node(text: str = "chunk text", paper_id: str = "p1"):
    node = MagicMock()
    node.metadata = {"paper_id": paper_id, "title": "T", "authors": "A", "year": 2024}
    node.get_content.return_value = text
    return node


def test_retrieve_nodes_vector_only_never_calls_bm25(monkeypatch):
    """use_hybrid=False must touch only the vector retriever, never BM25."""
    from research_assistant.qa import engine

    mock_nodes = [_make_mock_node()]
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = mock_nodes
    mock_index = MagicMock()
    mock_index.as_retriever.return_value = mock_retriever

    monkeypatch.setattr(engine.store, "get_index", lambda: mock_index)
    get_nodes_called = []
    monkeypatch.setattr(
        engine.store, "get_all_text_nodes", lambda: get_nodes_called.append(1) or []
    )

    result = engine._retrieve_nodes("question", wide_k=5, use_hybrid=False)

    assert result == mock_nodes
    mock_index.as_retriever.assert_called_once_with(similarity_top_k=5)
    assert get_nodes_called == [], "BM25 path must not be entered for use_hybrid=False"


def test_retrieve_nodes_hybrid_returns_empty_when_library_empty(monkeypatch):
    """Hybrid path short-circuits when there are no stored chunks."""
    from research_assistant.qa import engine

    monkeypatch.setattr(engine.store, "get_index", lambda: MagicMock())
    monkeypatch.setattr(engine.store, "get_all_text_nodes", lambda: [])

    result = engine._retrieve_nodes("question", wide_k=5, use_hybrid=True)

    assert result == []


def test_retrieve_nodes_hybrid_uses_fusion(monkeypatch):
    """Hybrid path builds both a BM25 and a vector retriever, fused with RRF.

    _import_hybrid_deps is monkeypatched so the test passes even before the
    llama-index-retrievers-bm25 package is installed.
    """
    from research_assistant.qa import engine

    stored_nodes = [_make_mock_node("n1", "p1"), _make_mock_node("n2", "p2")]
    fused_result = [_make_mock_node("fused", "p1")]

    mock_vec_retriever = MagicMock()
    mock_index = MagicMock()
    mock_index.as_retriever.return_value = mock_vec_retriever

    monkeypatch.setattr(engine.store, "get_index", lambda: mock_index)
    monkeypatch.setattr(engine.store, "get_all_text_nodes", lambda: stored_nodes)

    MockBM25Class = MagicMock()
    MockFusionClass = MagicMock()
    mock_bm25_instance = MagicMock()
    mock_fusion_instance = MagicMock()
    mock_fusion_instance.retrieve.return_value = fused_result
    MockBM25Class.from_defaults.return_value = mock_bm25_instance
    MockFusionClass.return_value = mock_fusion_instance

    monkeypatch.setattr(engine, "_import_hybrid_deps", lambda: (MockBM25Class, MockFusionClass))

    result = engine._retrieve_nodes("attention mechanism", wide_k=10, use_hybrid=True)

    MockBM25Class.from_defaults.assert_called_once_with(
        nodes=stored_nodes, similarity_top_k=10
    )
    mock_index.as_retriever.assert_called_once_with(similarity_top_k=10)

    assert MockFusionClass.call_count == 1
    _, kw = MockFusionClass.call_args
    assert kw["num_queries"] == 1
    assert kw["mode"] == "reciprocal_rerank"

    mock_fusion_instance.retrieve.assert_called_once_with("attention mechanism")
    assert result == fused_result


# ---------------------------------------------------------------------------
# ask(): high-level branching without network or model calls
# ---------------------------------------------------------------------------


def test_ask_returns_early_with_no_api_key(monkeypatch):
    from research_assistant.qa import engine
    from research_assistant.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", None)
    answer = engine.ask("any question")
    assert "ANTHROPIC_API_KEY" in answer.text
    assert answer.citations == []


def test_ask_returns_library_empty_message(monkeypatch):
    from research_assistant.qa import engine
    from research_assistant.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "sk-fake")
    monkeypatch.setattr(engine, "_retrieve_nodes", lambda *a, **kw: [])

    answer = engine.ask("anything", use_hybrid=False, use_reranker=False)
    assert "empty" in answer.text.lower()
    assert answer.citations == []


def test_ask_truncates_to_top_k_without_reranker(monkeypatch):
    """Without reranker the node list must be sliced to exactly top_k."""
    from research_assistant.qa import engine
    from research_assistant.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "sk-fake")

    many_nodes = [_make_mock_node(f"chunk {i}", f"p{i}") for i in range(10)]
    monkeypatch.setattr(engine, "_retrieve_nodes", lambda *a, **kw: many_nodes)

    context_sent: list[str] = []

    def fake_create(**kwargs):
        context_sent.append(kwargs["messages"][0]["content"])
        resp = MagicMock()
        resp.content = [MagicMock(type="text", text="Answer.")]
        return resp

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = fake_create
    monkeypatch.setattr(engine.anthropic, "Anthropic", lambda **kw: mock_client)

    answer = engine.ask("Q", top_k=3, use_hybrid=False, use_reranker=False)

    # 3 context blocks separated by "---" means exactly 2 separators in prompt.
    context_section = context_sent[0].split("Question:")[0]
    assert context_section.count("---") == 2
    assert len(answer.citations) == 3


def test_ask_calls_reranker_when_enabled(monkeypatch):
    """When use_reranker=True, _rerank must be called before passing nodes to Claude."""
    from research_assistant.qa import engine
    from research_assistant.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "sk-fake")

    raw_nodes = [_make_mock_node(f"raw {i}", f"p{i}") for i in range(5)]
    reranked_nodes = [_make_mock_node("best", "p2")]

    monkeypatch.setattr(engine, "_retrieve_nodes", lambda *a, **kw: raw_nodes)

    rerank_called_with: list = []

    def fake_rerank(nodes, question, top_n):
        rerank_called_with.append((nodes, question, top_n))
        return reranked_nodes

    monkeypatch.setattr(engine, "_rerank", fake_rerank)

    context_sent: list[str] = []

    def fake_create(**kwargs):
        context_sent.append(kwargs["messages"][0]["content"])
        resp = MagicMock()
        resp.content = [MagicMock(type="text", text="Reranked answer.")]
        return resp

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = fake_create
    monkeypatch.setattr(engine.anthropic, "Anthropic", lambda **kw: mock_client)

    answer = engine.ask("Q", top_k=3, use_hybrid=False, use_reranker=True)

    assert len(rerank_called_with) == 1
    nodes_arg, q_arg, top_n_arg = rerank_called_with[0]
    assert nodes_arg is raw_nodes
    assert q_arg == "Q"
    assert top_n_arg == 3
    # Only the reranked (1) node should appear in the prompt.
    assert context_sent[0].count("---") == 0
    assert len(answer.citations) == 1
