# Repository Guidelines

## Project Structure & Module Organization
- `summarize.py` is the main CLI entry point and LangGraph workflow (also exports `run_summary` + `TraceCollector` for the UI).
- `app.py` is the Streamlit UI for uploading a file and viewing a streaming workflow trace.
- `requirements.txt` lists Python dependencies.
- Sample inputs live at the repo root (e.g., `Software License Agreement.docx`, `Synthetic_SaaS_Software_License_Agreement_CZ_Law_FINAL.pdf`).
- `workflow.png` is a generated diagram of the LangGraph workflow (via `--graph`).
- Local config files: `.env.example` (template) and `.env` (local secrets; do not commit).

## Build, Test, and Development Commands
- `python -m venv .venv` creates the local virtual environment.
- `source .venv/bin/activate` activates the environment.
- `pip install -r requirements.txt` installs dependencies.
- `python summarize.py "Software License Agreement.docx"` runs a summary on a file.
- `streamlit run app.py` starts the UI.
- Optional CLI flags: `--raw-text`, `--show-chunks`, `--metrics`, `--graph PATH`, `--chunk-size N`, `--chunk-overlap N`.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and type hints (see `SummaryState`).
- Prefer descriptive, snake_case function names (e.g., `split_docs`).
- Keep prompts and CLI flags explicit and documented in the parser.
- No formatter or linter is configured; if adding one, align with PEP 8.

## Testing Guidelines
- No automated tests are currently defined.
- If adding tests, place them under a new `tests/` directory and use `test_*.py` naming.
- Prefer lightweight integration tests that run `summarize.py` with a small fixture file.

## Commit & Pull Request Guidelines
- No Git history is available in this directory, so no established commit convention exists.
- Use concise, imperative commit subjects (e.g., "Add PDF chunk metrics").
- PRs should include: a short description, any new CLI flags, and example command output.
- If changes affect prompts or outputs, include a before/after snippet in the PR.

## Security & Configuration Tips
- Keep API keys in `.env` only; never commit secrets.
- Set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` before running locally.
- LangSmith is optional; if enabled, set `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT`.
