"""Local query rewriting for academic search.

Converts a natural-language research question into 2-3 concise keyword
phrases optimised for the keyword-matching APIs used by arXiv, Semantic
Scholar, and OpenAlex (which work much better with "adolescent attention
span digital media" than with "what are the effects of screens on teenage
focus over the past decade?").

Model: Qwen/Qwen2.5-0.5B-Instruct (≈500 MB, runs on CPU, cached after the
first download).  The embedding model (Qwen3-Embedding) stays on the GPU if
available; this tiny generative model runs on CPU to avoid memory conflicts.
"""

from __future__ import annotations

import re

from research_assistant.config import settings

_pipeline: object | None = None   # process-lifetime cache

_SYSTEM = (
    "Convert the research question below into exactly 3 short keyword search "
    "queries for academic databases like Semantic Scholar and OpenAlex. "
    "Each query must be 3-6 words. "
    "Output only the 3 queries, one per line, with no numbering, no bullets, "
    "and no other text."
)

_STRIP_PREFIX = re.compile(r"^(?:\d+[.)]\s*|[-•*]\s*|[Qq]uery\s*\d*[:.]\s*|\")")
_STRIP_SUFFIX = re.compile(r"\"$")


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline  # type: ignore[import-untyped]

        _pipeline = pipeline(
            "text-generation",
            model=settings.query_rewriter_model,
            device=-1,   # CPU — keeps 2 GB GPU free for embeddings
            dtype="auto",
        )
    return _pipeline


def rewrite(topic: str) -> list[str]:
    """Return 2-3 keyword-optimised search queries for *topic*.

    Falls back to ``[topic]`` if the model fails or produces no usable output,
    so callers never have to handle errors from this function.
    """
    try:
        pipe = _get_pipeline()
        output = pipe(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": topic},
            ],
            max_new_tokens=80,
            do_sample=False,
        )
        raw: str = output[0]["generated_text"][-1]["content"]

        queries: list[str] = []
        for line in raw.strip().splitlines():
            line = _STRIP_PREFIX.sub("", line.strip())
            line = _STRIP_SUFFIX.sub("", line).strip()
            if line:
                queries.append(line)

        queries = queries[:3]
        return queries if queries else [topic]

    except Exception:  # noqa: BLE001 — never crash the caller
        return [topic]
