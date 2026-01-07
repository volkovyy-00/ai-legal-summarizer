# Architecture

## High-level overview
This project is a single-process Python application with two entrypoints:

- **CLI** (`summarize.py`) for summarizing a local file path.
- **Streamlit UI** (`app.py`) for uploading a file and viewing a summary and a live-updating trace.

Both entrypoints call the same core function: `summarize.run_summary(...)`.

At runtime, `run_summary()` builds a small **LangGraph** graph that orchestrates:
1) document load
2) chunking
3) per-chunk summarization (map)
4) final reduction (reduce)

## Components and responsibilities

### `summarize.py`
Primary module that contains:
- `SummaryState`: the LangGraph workflow state type (a `TypedDict`).
- `TraceCollector`: collects step-level and LLM-call timing data and can emit events to a UI callback.
- `_build_graph(...)`: builds and compiles the LangGraph `StateGraph`.
- `run_summary(...)`: public API for running a summary (used by both CLI and UI).
- `main()`: CLI entry point and argument parsing.

#### Workflow nodes
- `load_docs`
  - Chooses a loader by extension:
    - `.docx` → `Docx2txtLoader`
    - `.pdf` → `PyPDFLoader`
  - Emits a trace step with the number of loaded documents/pages.
- `split_docs`
  - Concatenates all loaded documents into a single text blob (prevents “one chunk per page” behavior).
  - Splits into character-based chunks using `RecursiveCharacterTextSplitter` with overlap.
  - Emits a trace step with `chunks` and total `chars`.
- `map_summarize`
  - For each chunk: calls the LLM (sequentially) with a bullet-summary prompt.
  - Emits a trace event per LLM call, including duration and usage metadata (if provided).
- `reduce_summaries`
  - If more than one chunk: calls the LLM to combine/clean the chunk summaries.
  - Emits a trace step.

### `app.py` (Streamlit UI)
Responsibilities:
- Accept file upload (PDF/DOCX).
- Save to a temporary file path, call `run_summary()` with a `TraceCollector`.
- Render:
  - final summary
  - a collapsed “Trace” section with progress bar + tables
- Stream trace updates during the run by registering `TraceCollector(on_event=...)` and updating `st.empty()` containers.

### External services
- **OpenAI API** (required): used by `langchain_openai.ChatOpenAI`.
- **LangSmith** (optional): LangChain can emit traces when `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` is set.
- **Mermaid.ink** (optional): used by default to render `--graph` PNG (network-dependent).

## Interfaces between components
- UI → core: `from summarize import run_summary, TraceCollector`
- CLI → core: calls `run_summary()` in `main()`
- Trace streaming: `TraceCollector(on_event=callable)` emits dict events such as:
  - `{type: "step", name: "split_docs", duration: ..., metadata: {...}}`
  - `{type: "llm_call", stage: "map_summarize", chunk_index: 1, duration: ..., usage: {...}}`

There are no HTTP APIs, queues, or databases in the current design.

## Data model overview

### LangGraph state (`SummaryState`)
- `file_path`: input path (string)
- `documents`: list of loaded `Document` objects
- `splits`: list of chunk `Document` objects
- `chunk_summaries`: list of bullet summaries (strings)
- `final_summary`: string output

### Trace model (in-memory)
- `TraceCollector.steps`: list of step timing entries
- `TraceCollector.llm_calls`: list of LLM call entries
- `TraceCollector.events`: append-only event stream (used by Streamlit for live updates)

## Performance and reliability considerations
- **Latency drivers**
  - Map step makes 1 LLM call per chunk; calls are sequential.
  - Reduce step makes an additional call when `N_chunks > 1`.
- **Chunking tradeoff**
  - Larger `chunk_size` → fewer calls → lower latency, but must remain within model context window.
  - Overlap increases total tokens and can increase latency/cost.
- **Failure modes**
  - Missing API key, unsupported file type, or unextractable PDF → fail fast.
  - OpenAI/network errors currently bubble up (no retry/backoff).

## Tradeoffs and alternatives considered
- **LangGraph vs direct function calls**
  - LangGraph makes the pipeline explicit and easy to visualize.
  - For this demo, it is somewhat heavier than necessary, but helpful for teaching/traceability.
- **Map-reduce summarization vs single-shot**
  - Map-reduce is robust for long texts but increases latency due to multiple calls.
  - A future single-shot path could be added when input fits within the model context.
- **Local trace vs LangSmith-only**
  - Local trace gives immediate feedback in the UI with no external dependency.
  - LangSmith remains useful for deeper inspection and persistence.

