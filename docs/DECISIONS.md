# Design decisions (ADR-style notes)

## 1) Use LangChain + OpenAI for summarization
**Decision**
- Use `langchain_openai.ChatOpenAI` for the LLM calls.

**Why**
- Minimal integration effort, widely used interface.

**Tradeoffs**
- Strong dependency on external API availability and network latency.

## 2) Use LangGraph to orchestrate the workflow
**Decision**
- Model the pipeline as a LangGraph `StateGraph` with explicit nodes and edges.

**Why**
- Makes the workflow structure explicit and visualizable.
- Helps teach agent/workflow concepts.

**Tradeoffs**
- More abstraction than a plain function pipeline for a small demo.

## 3) Use map-reduce summarization
**Decision**
- Summarize each chunk (map) and then combine into a final summary (reduce).

**Why**
- Handles long contracts that may exceed a model’s context window.

**Tradeoffs**
- Increases latency and cost vs single-shot summarization.

## 4) Concatenate pages before splitting
**Decision**
- Merge all loaded documents/pages into one string, then split.

**Why**
- Avoids creating one “chunk” per PDF page (which increases the number of LLM calls).

**Tradeoffs**
- Loses per-page boundaries (acceptable for this demo).

## 5) Add local trace collection + streaming UI updates
**Decision**
- Implement `TraceCollector` to capture step timings and per-call timings and stream events to Streamlit.

**Why**
- Allows fast diagnosis of “where time goes” without requiring LangSmith.

**Tradeoffs**
- Not a full execution trace (no token-level streaming, no persistence).

## 6) Keep LangSmith tracing optional via env vars
**Decision**
- Support LangSmith via standard `LANGCHAIN_*` environment variables.

**Why**
- Enables deeper debugging and historical traces when desired.

**Tradeoffs**
- Sends data externally; users must opt-in and understand privacy implications.

## 7) Keep dependencies minimal and unpinned
**Decision**
- Use `requirements.txt` with lower bounds (e.g., `>=`) rather than pinned versions.

**Why**
- Easier onboarding for a demo.

**Tradeoffs**
- Reduced reproducibility; future installs may behave differently.

