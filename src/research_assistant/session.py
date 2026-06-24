"""Tiny on-disk session state.

The CLI starts a fresh process every time you run it, so "remembering" the
results of the last `research search` (so `research save 2` can refer to
result #2) needs to live on disk rather than in memory. This module is that
one small piece of persistence.
"""

from __future__ import annotations

import json

from research_assistant.config import settings
from research_assistant.sources.base import Paper

_LAST_SEARCH_PATH = settings.data_dir / "last_search.json"


def save_last_search(papers: list[Paper]) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    _LAST_SEARCH_PATH.write_text(
        json.dumps([p.to_dict() for p in papers], indent=2), encoding="utf-8"
    )


def load_last_search() -> list[Paper]:
    if not _LAST_SEARCH_PATH.exists():
        return []
    data = json.loads(_LAST_SEARCH_PATH.read_text(encoding="utf-8"))
    return [Paper.from_dict(item) for item in data]


def resolve_paper_ref(ref: str) -> Paper | None:
    """Resolve a CLI argument to a `Paper`.

    `ref` may be either:
      - a 1-based index into the most recent `search` results (e.g. "2"), or
      - a full paper id (e.g. "arxiv:2301.12345")
    """
    papers = load_last_search()

    if ref.isdigit():
        index = int(ref) - 1
        if 0 <= index < len(papers):
            return papers[index]
        return None

    for paper in papers:
        if paper.id == ref:
            return paper
    return None
