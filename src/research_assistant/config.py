"""Central configuration for the research assistant.

Loads values from a `.env` file (via python-dotenv) and from the real
environment, with sensible defaults for everything else. Import `settings`
from this module wherever you need configuration -- don't read
`os.environ` directly elsewhere, so there's exactly one place that knows
how configuration is resolved.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if one exists.
# This does NOT override variables that are already set in the real
# environment (e.g. exported in your shell) -- env vars always win.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _mask(value: str | None) -> str:
    """Return a value safe to print/log -- secrets are never shown in full."""
    if not value:
        return "<not set>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


class Settings:
    """Resolved configuration for this run of the application."""

    def __init__(self) -> None:
        # --- Anthropic (reasoning / synthesis LLM, Phase 3+) ---
        self.anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
        self.reasoning_model: str = os.getenv("REASONING_MODEL", "claude-haiku-4-5")

        # --- Local embeddings (retrieval, Phase 2+) ---
        # The dedicated Qwen3 *embedding* model -- not the Qwen3 chat model.
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
        )

        # --- Optional API key for higher Semantic Scholar rate limits ---
        # The Semantic Scholar API works without a key at a lower rate limit;
        # set S2_API_KEY in .env if you have one and hit limits.
        self.semantic_scholar_api_key: str | None = os.getenv("S2_API_KEY")

        # --- Phase 5.5: query rewriting (local generative model) ---
        # A small instruction-tuned model that converts natural-language questions
        # into 2-3 tight keyword phrases before hitting live paper APIs.
        self.query_rewriter_model: str = os.getenv(
            "QUERY_REWRITER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
        )

        # --- Phase 4: retrieval quality (hybrid search + reranking) ---
        # Cross-encoder model used to rerank retrieved chunks. Any HuggingFace
        # cross-encoder compatible with sentence-transformers works here.
        self.reranker_model: str = os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        # How many chunks to keep after reranking (passed to Claude).
        self.reranker_top_n: int = int(os.getenv("RERANKER_TOP_N", "6"))
        # Wide retrieval pool before reranking; larger = better recall, slower rerank.
        self.retrieval_wide_k: int = int(os.getenv("RETRIEVAL_WIDE_K", "20"))

        # --- Optional contact email for OpenAlex's "polite pool" ---
        # Setting this gets faster, less-throttled responses. Optional.
        self.contact_email: str | None = os.getenv("CONTACT_EMAIL")

        # --- Storage ---
        self.data_dir: Path = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
        self.chroma_dir: Path = self.data_dir / "chroma"
        self.collection_name: str = os.getenv("CHROMA_COLLECTION", "papers")

        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    def redacted(self) -> dict[str, str]:
        """A dict safe to print or log -- secrets are masked."""
        return {
            "anthropic_api_key": _mask(self.anthropic_api_key),
            "reasoning_model": self.reasoning_model,
            "embedding_model": self.embedding_model,
            "query_rewriter_model": self.query_rewriter_model,
            "reranker_model": self.reranker_model,
            "reranker_top_n": str(self.reranker_top_n),
            "retrieval_wide_k": str(self.retrieval_wide_k),
            "semantic_scholar_api_key": _mask(self.semantic_scholar_api_key),
            "contact_email": self.contact_email or "<not set>",
            "data_dir": str(self.data_dir),
            "chroma_dir": str(self.chroma_dir),
            "collection_name": self.collection_name,
        }

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.redacted().items())
        return f"Settings({fields})"


# Module-level singleton -- created once on first import, reused everywhere.
settings = Settings()
