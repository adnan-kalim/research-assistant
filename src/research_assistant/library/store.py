"""Local vector store for the personal paper library.

Wraps a LlamaIndex `VectorStoreIndex` backed by a persistent Chroma
collection, embedding text with a local Qwen3-Embedding model. Everything
here runs on your machine -- no network calls, no per-query cost. This is
stages 2 and 3 of the RAG pipeline (EMBED, STORE) -- see README.md.
"""

from __future__ import annotations

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from research_assistant.config import settings

_embed_model: HuggingFaceEmbedding | None = None
_chroma_client: chromadb.ClientAPI | None = None
_index: VectorStoreIndex | None = None


def get_embed_model() -> HuggingFaceEmbedding:
    """Lazily load the local embedding model (downloads on first use, then cached)."""
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
    return _embed_model


def _get_collection():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return _chroma_client.get_or_create_collection(settings.collection_name)


def get_index() -> VectorStoreIndex:
    """The persistent vector index backing the library (lazily created)."""
    global _index
    if _index is not None:
        return _index

    vector_store = ChromaVectorStore(chroma_collection=_get_collection())
    _index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=get_embed_model(),
    )
    return _index


def chunk_count() -> int:
    """Number of stored chunks. A single paper is usually split into several."""
    return _get_collection().count()


def list_saved_papers() -> list[dict]:
    """One row per distinct saved paper, reconstructed from chunk metadata."""
    result = _get_collection().get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in result.get("metadatas") or []:
        paper_id = meta.get("paper_id") if meta else None
        if paper_id and paper_id not in seen:
            seen[paper_id] = meta
    return list(seen.values())


def delete_paper(paper_id: str) -> None:
    """Remove all chunks for `paper_id`, if any.

    Called before re-saving a paper so re-running `save` never creates
    duplicate chunks for the same paper.
    """
    _get_collection().delete(where={"paper_id": paper_id})
