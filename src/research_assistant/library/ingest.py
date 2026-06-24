"""Save a Paper into the personal library: chunk -> embed -> upsert.

This is stage 1 (INGEST) of the RAG pipeline -- see README.md. We always
have title + abstract to work with; full-text PDF extraction is attempted
on a best-effort basis and silently skipped if it fails, so a flaky
download never blocks saving a paper.
"""

from __future__ import annotations

from io import BytesIO

import requests
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from research_assistant.library import store
from research_assistant.sources.base import Paper

_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64
_splitter = SentenceSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)


def _try_fetch_full_text(paper: Paper) -> str | None:
    """Best-effort full-text extraction from the paper's PDF.

    Returns None on any failure (no PDF available, network error,
    unparseable file) -- callers fall back to title + abstract, which is
    always available.
    """
    if not paper.pdf_url:
        return None
    try:
        import pypdf

        response = requests.get(paper.pdf_url, timeout=30)
        response.raise_for_status()
        reader = pypdf.PdfReader(BytesIO(response.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() or None
    except Exception:
        return None


def save_paper(paper: Paper, full_text: bool = False) -> int:
    """Embed and store `paper` in the library. Returns the number of chunks written.

    Defaults to title + abstract only, which is fast (a few seconds) and
    enough for most questions. Full-text PDF extraction (`full_text=True`)
    gives the Q&A engine much richer material to ground answers in, but
    embedding a whole paper's worth of chunks on CPU can take several
    minutes -- opt into it per paper when you want that depth.

    Safe to call again for an already-saved paper: existing chunks for that
    paper id are removed first, so re-saving never creates duplicates.
    """
    store.delete_paper(paper.id)

    text_parts = [paper.title]
    if paper.abstract:
        text_parts.append(paper.abstract)
    if full_text:
        body = _try_fetch_full_text(paper)
        if body:
            text_parts.append(body)

    document = Document(
        text="\n\n".join(text_parts),
        metadata={
            "paper_id": paper.id,
            "title": paper.title,
            "authors": ", ".join(paper.authors),
            "year": paper.year or 0,
            "source": paper.source,
            "url": paper.url or "",
        },
    )

    nodes = _splitter.get_nodes_from_documents([document])
    store.get_index().insert_nodes(nodes)
    return len(nodes)
