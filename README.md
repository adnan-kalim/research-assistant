# AI-Powered Research Assistant

Search millions of scholarly papers, build a personal library, ask grounded questions with citations, and generate structured literature reviews — all from the command line.

See `DOCS.md` for full service and CLI documentation.

---

## Setup

### 1. Create a virtual environment and install dependencies

```sh
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
pip install -e .              # registers the `research` CLI command
```

> **GPU note:** `requirements.txt` installs the CPU-only PyTorch wheel (~200 MB).
> Remove the `--extra-index-url` line if you have a CUDA GPU with 6+ GB VRAM.

### 2. Set your Anthropic API key

```sh
copy .env.example .env    # Windows
# cp .env.example .env    # macOS / Linux
```

Edit `.env` and fill in `ANTHROPIC_API_KEY`. The `ask` and `review` commands need it; `search` and `save` are free.

### 3. Verify the setup

```sh
python -c "from research_assistant.config import settings; print(settings)"
```

Should print all config values with the API key masked.

---

## Quick start

```sh
research search "attention is all you need"   # discover papers live
research save 1                               # save result #1 to your library
research ask "How does multi-head attention work?"  # cited Q&A over your library
research review "transformer architectures"   # full literature review
```

---

## Running tests

```sh
pytest
```

All tests mock network and model calls — they run in seconds with no GPU or internet required.
