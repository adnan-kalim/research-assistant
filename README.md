# research-assistant

An AI-powered research assistant with access to scholarly papers: search live
academic indexes (arXiv, Semantic Scholar, OpenAlex), save papers to a
personal library, ask grounded questions answered with citations, and
synthesize literature reviews across many papers.

See `PLAN.md` for the full build plan and phased roadmap.

## How it works (the 5-stage RAG pipeline)

```
1. INGEST    grab paper text (title, abstract, full text)
2. EMBED     turn text into vectors that capture meaning      -> Qwen3-Embedding (local)
3. STORE     save vectors in a vector DB, searchable by similarity -> Chroma (local)
4. RETRIEVE  embed the question, find the most similar chunks
5. GENERATE  hand those chunks + the question to an LLM        -> Claude Haiku 4.5
```

## Setup

1. Create and activate a virtual environment, then install dependencies:

   ```sh
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   pip install -r requirements.txt
   pip install -e .                # installs the `research` CLI command
   ```

2. Copy `.env.example` to `.env` and fill in your Anthropic API key:

   ```sh
   copy .env.example .env
   ```

3. Verify configuration loads correctly:

   ```sh
   python -c "from research_assistant.config import settings; print(settings)"
   ```

## Usage

```sh
research search "graph neural networks"     # live search across arXiv etc.
research save <result-number-or-id>          # save a paper to your local library
research library                             # list saved papers
research ask "your question"                 # ask a cited question over your library
research review "a topic"                    # synthesize a literature review (Phase 5)
```

## Project layout

```
src/research_assistant/
  config.py       # env/config loading
  sources/        # live API clients (arXiv, Semantic Scholar, OpenAlex)
  library/        # personal saved-paper store (Chroma + Qwen3-Embedding)
  qa/             # retrieval + Claude Haiku 4.5 cited Q&A
  synthesis/      # multi-paper literature review synthesis
  cli.py          # command-line entry point
  api/            # FastAPI surface (Phase 6)
```
