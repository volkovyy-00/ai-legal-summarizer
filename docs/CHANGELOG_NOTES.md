# Changelog notes (current behavior + known gaps)

## Current behavior
- Summarizes `.docx` and `.pdf` contracts into bullet points using OpenAI via LangChain.
- Orchestrates the pipeline using LangGraph:
  - load → merge/split → map summarize → reduce
- CLI supports:
  - `--raw-text` to print extracted text
  - `--show-chunks` to print intermediate chunk summaries
  - `--metrics` to print elapsed time and token usage (when reported)
  - `--graph PATH` to write a PNG visualization of the graph (may require network)
  - `--chunk-size` / `--chunk-overlap` to tune chunking behavior
- Streamlit UI (`app.py`) supports:
  - uploading a file
  - rendering the final summary
  - a collapsed “Trace” section with streaming progress updates and timing tables
- LangSmith tracing can be enabled via `.env` variables.

## Known gaps / risks
- No OCR: scanned PDFs can fail or yield poor text.
- No retries/backoff: OpenAI rate limits/timeouts bubble up to the user.
- No parallelism: chunk summarization is sequential; performance depends heavily on API latency.
- No automated tests or CI.
- Dependencies are not pinned; environment drift is possible.
- Graph PNG rendering defaults to Mermaid.ink and may fail under restricted network policies.

## Suggested next improvements (not implemented)
- Add a “single-shot” mode when the full text fits in the model context window.
- Add exponential backoff retries for transient OpenAI errors.
- Add basic unit/integration tests with a mocked LLM.
- Add optional OCR pipeline for scanned PDFs.
- Add local graph rendering option (no network dependency).

