"""Domain-aware source routing for live paper search.

Routes search queries to the scholarly sources most likely to have relevant
papers. The core decision is almost always "include arXiv or not" -- arXiv is
STEM-only, while Semantic Scholar and OpenAlex are multidisciplinary. A social
science or psychology query sent to arXiv gets astrophysics papers that happen
to contain the word "attention" or "ten years".

Routing uses local embedding-based zero-shot classification:
  1. Each domain has a short descriptive phrase (DOMAIN_DESCRIPTIONS).
  2. On first use those phrases are embedded with the already-loaded
     Qwen3-Embedding model and cached for the process lifetime.
  3. At query time: embed the topic → cosine similarity against each cached
     domain vector → pick the closest → map to that domain's source list.

This reuses store.get_embed_model() -- the same cached model used for paper
ingestion. No extra download, no API key, no per-call cost.

Priority in resolve_sources():
  1. Explicit --sources flag  → used as-is, no classification.
  2. Explicit --domain preset → map to source list, no classification.
  3. --domain auto (default)  → classify_domain() → map to source list.
"""

from __future__ import annotations

import numpy as np

from research_assistant.library import store

# ---------------------------------------------------------------------------
# Domain → source mapping
# ---------------------------------------------------------------------------

DOMAIN_SOURCES: dict[str, list[str]] = {
    "cs":             ["arxiv", "semantic_scholar"],
    "physics":        ["arxiv", "semantic_scholar", "openalex"],
    "math":           ["arxiv", "semantic_scholar", "openalex"],
    "economics":      ["arxiv", "semantic_scholar", "openalex"],
    "social_science": ["semantic_scholar", "openalex"],
    "psychology":     ["semantic_scholar", "openalex"],
    "medicine":       ["semantic_scholar", "openalex"],
    "biology":        ["semantic_scholar", "openalex"],
    "humanities":     ["semantic_scholar", "openalex"],
    "all":            ["arxiv", "semantic_scholar", "openalex"],
}

# Short keyword-rich descriptions used as classification targets.
# "all" is intentionally omitted -- it is a manual escape hatch, never
# auto-detected.
DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "cs": (
        "computer science machine learning artificial intelligence algorithms "
        "software engineering programming neural networks deep learning data structures "
        "natural language processing computer vision robotics"
    ),
    "physics": (
        "physics quantum mechanics thermodynamics electromagnetism astrophysics "
        "cosmology particle physics condensed matter optics relativity atomic physics "
        "nuclear physics supernovae gravitational waves"
    ),
    "math": (
        "mathematics algebra geometry topology number theory statistics probability "
        "calculus mathematical analysis combinatorics discrete mathematics proofs"
    ),
    "economics": (
        "economics econometrics finance markets game theory macroeconomics "
        "microeconomics trade monetary policy labour market fiscal policy growth"
    ),
    "social_science": (
        "sociology anthropology political science social behavior cultural studies "
        "public policy demographics inequality social media online communities "
        "civic engagement governance"
    ),
    "psychology": (
        "psychology cognitive science mental health behavior neuroscience attention "
        "memory emotion child development adolescent learning motivation cognition "
        "ADHD executive function screen time"
    ),
    "medicine": (
        "medicine clinical research disease health pharmacology epidemiology surgery "
        "therapy patient outcomes diagnosis treatment public health nursing"
    ),
    "biology": (
        "biology genetics evolution ecology cell biology molecular biology biochemistry "
        "microbiology neurobiology genomics protein metabolism"
    ),
    "humanities": (
        "history philosophy literature linguistics art music religion ethics "
        "cultural history archaeology ancient civilizations language"
    ),
}

# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

_domain_vectors: dict[str, list[float]] | None = None


def _domain_embeddings() -> dict[str, list[float]]:
    """Embed domain descriptions once and cache them for the process lifetime."""
    global _domain_vectors
    if _domain_vectors is not None:
        return _domain_vectors
    embed_model = store.get_embed_model()
    _domain_vectors = {
        domain: embed_model.get_text_embedding(description)
        for domain, description in DOMAIN_DESCRIPTIONS.items()
    }
    return _domain_vectors


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_domain(topic: str) -> str:
    """Return the domain whose description is most similar to *topic*.

    Uses cosine similarity of Qwen3-Embedding vectors. The embedding model is
    loaded lazily on first call (already cached if any paper has been saved).
    """
    embed_model = store.get_embed_model()
    topic_vec = embed_model.get_text_embedding(topic)
    best_domain, best_score = "all", -1.0
    for domain, vec in _domain_embeddings().items():
        score = _cosine_similarity(topic_vec, vec)
        if score > best_score:
            best_score, best_domain = score, domain
    return best_domain


def resolve_sources(
    domain: str | None,
    sources: str | None,
    topic: str,
) -> tuple[list[str], str]:
    """Return *(source_names, explanation)* following the priority chain.

    Args:
        domain:   ``"auto"`` / ``None`` for auto-detect, or a preset key from
                  DOMAIN_SOURCES (e.g. ``"psychology"``), or ``"all"``.
        sources:  Comma-separated explicit source names (e.g. ``"arxiv,openalex"``).
                  Overrides *domain* entirely when provided.
        topic:    The search query / review topic (used only for auto-detect).

    The *explanation* string describes what was chosen and why, suitable for
    printing in the CLI so the user knows which sources are being queried.
    """
    from research_assistant.sources.aggregator import ALL_SOURCES

    # Priority 1: explicit --sources
    if sources:
        names = [s.strip() for s in sources.split(",") if s.strip()]
        unknown = [n for n in names if n not in ALL_SOURCES]
        valid = [n for n in names if n in ALL_SOURCES]
        if unknown:
            explanation = (
                f"explicit sources: {', '.join(valid) or 'none'}"
                f" (unknown ignored: {', '.join(unknown)})"
            )
        else:
            explanation = f"explicit sources: {', '.join(valid)}"
        return valid or list(ALL_SOURCES), explanation

    # Priority 2: explicit --domain preset (not "auto")
    if domain and domain != "auto":
        if domain not in DOMAIN_SOURCES:
            return list(ALL_SOURCES), f"unknown domain {domain!r} — querying all sources"
        names = DOMAIN_SOURCES[domain]
        return names, f"domain: {domain} → {', '.join(names)}"

    # Priority 3: auto-detect via embedding similarity
    detected = classify_domain(topic)
    names = DOMAIN_SOURCES[detected]
    excluded = [s for s in ALL_SOURCES if s not in names]
    explanation = f"auto-detected domain: {detected}"
    if excluded:
        explanation += f" ({', '.join(excluded)} excluded)"
    return names, explanation
