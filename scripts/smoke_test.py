"""End-to-end smoke test for Phases 0-3: search -> save -> ask.

Run with:
    python scripts/smoke_test.py

Exits non-zero on failure so it's usable in automation later, not just
for a human watching the output.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Fallback for running this script without `pip install -e .` having been
# done -- harmless if the package is already installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_assistant.library import ingest  # noqa: E402
from research_assistant.qa import engine as qa_engine  # noqa: E402
from research_assistant.sources.aggregator import search_all  # noqa: E402

QUERY = "transformer attention mechanism"
QUESTION = "What problem does the attention mechanism solve?"


def main() -> int:
    # Windows consoles often default to a legacy code page that can't render
    # non-ASCII author names/abstracts -- force UTF-8 if possible.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    print(f"1. Searching arXiv for: {QUERY!r}")
    papers, warnings = search_all(QUERY, limit=3, source_names=["arxiv"])
    for warning in warnings:
        print(f"   warning: {warning}")
    if not papers:
        print("   FAILED: no search results")
        return 1
    print(f"   got {len(papers)} result(s)")

    print("2. Saving results to the library (embedding runs locally)")
    saved = 0
    for paper in papers:
        chunks = ingest.save_paper(paper)
        print(f"   saved {paper.id} ({chunks} chunk(s))")
        saved += 1
    if saved == 0:
        print("   FAILED: nothing saved")
        return 1

    print(f"3. Asking: {QUESTION!r}")
    answer = qa_engine.ask(QUESTION)
    print("   --- answer ---")
    print("   " + answer.text.replace("\n", "\n   "))
    if answer.citations:
        print("   --- citations ---")
        for citation in answer.citations:
            print(f"   - {citation.title} ({citation.year}) -- {citation.authors}")
    else:
        print("   warning: no citations returned (check ANTHROPIC_API_KEY)")

    print("\nSmoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
