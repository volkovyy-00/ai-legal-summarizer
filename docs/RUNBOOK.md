# Runbook

## Start / stop

### CLI
Start:
```bash
source .venv/bin/activate
python summarize.py "Synthetic_SaaS_Software_License_Agreement_CZ_Law_FINAL.pdf"
```

Stop:
- Press `Ctrl+C` (or wait for the run to complete).

### Streamlit UI
Start:
```bash
source .venv/bin/activate
streamlit run app.py
```

Stop:
- Press `Ctrl+C` in the terminal where Streamlit is running.

## Health checks
- Environment:
  - `OPENAI_API_KEY` is set (in shell or `.env`).
  - `python -c "import streamlit, langchain_openai, langgraph"` succeeds.
- CLI smoke test:
  - Run `python summarize.py "Software License Agreement.docx" --metrics`
  - Confirm it prints a summary and metrics.
- UI smoke test:
  - Open the Streamlit page, upload a document, click **Summarize**.
  - Confirm summary renders and Trace updates during execution.

## Common failure modes

### Missing API key
**Symptom**
- CLI prints: `Missing OPENAI_API_KEY in environment or .env.`
- UI shows an error banner with the same message.

**Fix**
- Add `OPENAI_API_KEY=...` to `.env` (do not commit) or export it in your shell.

### Unsupported file type
**Symptom**
- Error: `Unsupported file type. Use .docx or .pdf.`

**Fix**
- Convert to `.pdf` or `.docx`.

### PDF loads but produces no text
**Symptom**
- Error: `No content loaded from the file.`

**Likely cause**
- The PDF is scanned/bitmap; this project does not do OCR.

**Fix**
- Use a text-based PDF or add OCR as a future enhancement.

### Slow runs / high latency
**Symptoms**
- Runs take tens of seconds or minutes.
- Trace shows many `map_summarize` LLM calls.

**Fixes**
- Increase `--chunk-size` to reduce number of chunks/calls.
- Reduce `--chunk-overlap` to lower repeated input.
- Prefer a smaller/faster model via `--model` / `OPENAI_MODEL`.
- Use the Streamlit Trace view to confirm where time is spent:
  - local splitting is fast
  - LLM calls dominate wall time

### Graph PNG generation fails
**Symptom**
- CLI warning: `Failed to render graph: ...`

**Likely cause**
- `draw_mermaid_png()` uses a network renderer by default (Mermaid.ink) which may be blocked.

**Fix**
- Ignore (summary still works), or render the graph with a local backend (not implemented).

## Safe rollback steps
There is no formal release process in this repo. Safe rollback options:
- If using Git: `git checkout <last-known-good-commit>`.
- If not using Git: restore previous versions of `summarize.py` / `app.py` from your backups.

## Incident notes template
Capture the following:
- Timestamp + environment (OS, Python version)
- Input type and approximate size (pages / bytes)
- Model name + temperature
- Chunk settings (`chunk_size`, `chunk_overlap`)
- Total chunks and total runtime (from Trace / `--metrics`)
- Error messages + stack traces (if any)
- Any OpenAI request IDs (if surfaced by the SDK)

## “What changed?” checklist (before sharing/demoing)
- Did the prompts change? (expect different summary style)
- Did chunking defaults change? (affects latency and completeness)
- Did the model change? (affects latency/cost/quality)
- Is tracing enabled? (`LANGCHAIN_TRACING_V2=true` sends data externally)
- Does `--graph` work in the current network environment?

