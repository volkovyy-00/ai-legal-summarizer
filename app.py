#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from summarize import TraceCollector, run_summary


def _save_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return handle.name


def _render_trace(trace: TraceCollector, trace_state: dict) -> None:
    st.subheader("Trace")
    chunks_total = trace_state.get("chunks_total")
    chunks_done = trace_state.get("chunks_done", 0)
    status = trace_state.get("status", "Idle")
    if chunks_total:
        progress = min(chunks_done / max(chunks_total, 1), 1.0)
        st.progress(progress)
        st.write(f"{status} ({chunks_done}/{chunks_total} chunks)")
    else:
        st.write(status)

    st.markdown("Steps")
    if not trace.steps:
        st.write("No step timing data available.")
    else:
        step_rows = []
        for step in trace.steps:
            row = {"step": step["name"], "duration_s": f"{step['duration']:.2f}"}
            row.update(step.get("metadata", {}))
            step_rows.append(row)
        st.table(step_rows)

    st.markdown("LLM calls")
    if not trace.llm_calls:
        st.write("No LLM call data available.")
        return

    total_calls = len(trace.llm_calls)
    total_llm_time = sum(call["duration"] for call in trace.llm_calls)
    total_input = sum(call.get("usage", {}).get("input_tokens", 0) for call in trace.llm_calls)
    total_output = sum(
        call.get("usage", {}).get("output_tokens", 0) for call in trace.llm_calls
    )
    st.write(
        f"Calls: {total_calls} | Total LLM time: {total_llm_time:.2f}s | "
        f"Tokens in/out: {total_input}/{total_output}"
    )

    call_rows = []
    for call in trace.llm_calls:
        row = {
            "stage": call["stage"],
            "duration_s": f"{call['duration']:.2f}",
        }
        if "chunk_index" in call:
            row["chunk"] = call["chunk_index"]
        usage = call.get("usage", {})
        if usage:
            row["input_tokens"] = usage.get("input_tokens", 0)
            row["output_tokens"] = usage.get("output_tokens", 0)
        call_rows.append(row)
    st.dataframe(call_rows, use_container_width=True)


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="AI Legal Summarizer", layout="wide")
    st.title("AI Legal Summarizer")
    st.write("Upload a .pdf or .docx file to generate a summary.")

    uploaded = st.file_uploader("Upload a file", type=["pdf", "docx"])
    run_clicked = st.button("Summarize", type="primary", disabled=uploaded is None)

    trace_expander = st.expander("Trace", expanded=False)
    with trace_expander:
        trace_placeholder = st.empty()

    if not run_clicked or uploaded is None:
        with trace_placeholder.container():
            st.write("Trace will appear here after you run a summary.")
        return

    trace_state = {
        "status": "Starting",
        "chunks_total": None,
        "chunks_done": 0,
    }
    trace_ref = {"trace": None}

    def render_trace() -> None:
        if trace_ref["trace"] is None:
            return
        with trace_placeholder.container():
            _render_trace(trace_ref["trace"], trace_state)

    def on_trace_event(event: dict) -> None:
        event_type = event.get("type")
        if event_type == "step":
            name = event.get("name")
            if name == "load_docs":
                trace_state["status"] = "Loaded documents"
            elif name == "split_docs":
                trace_state["status"] = "Split into chunks"
                chunks = event.get("metadata", {}).get("chunks")
                if isinstance(chunks, int):
                    trace_state["chunks_total"] = chunks
            elif name == "map_summarize":
                trace_state["status"] = "Summarized chunks"
            elif name == "reduce_summaries":
                trace_state["status"] = "Reduced summaries"
        elif event_type == "llm_call":
            stage = event.get("stage")
            if stage == "map_summarize":
                trace_state["status"] = "Summarizing chunks"
                chunk_index = event.get("chunk_index")
                if isinstance(chunk_index, int):
                    trace_state["chunks_done"] = max(
                        trace_state["chunks_done"], chunk_index
                    )
            elif stage == "reduce_summaries":
                trace_state["status"] = "Reducing summaries"
        render_trace()

    trace = TraceCollector(on_event=on_trace_event)
    trace_ref["trace"] = trace

    temp_path = _save_upload(uploaded)
    try:
        with st.spinner("Summarizing..."):
            result = run_summary(file_path=temp_path, trace=trace)
    except Exception as exc:
        st.error(str(exc))
        return
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    st.subheader("Summary")
    st.markdown(result["summary"])

    trace_state["status"] = "Done"
    render_trace()


if __name__ == "__main__":
    main()
