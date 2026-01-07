#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph


class SummaryState(TypedDict, total=False):
    """Typed state for the LangGraph workflow."""

    file_path: str
    documents: List[Document]
    splits: List[Document]
    chunk_summaries: List[str]
    final_summary: str


class TraceCollector:
    """Collect step and LLM call timing data for UI tracing."""

    def __init__(self, on_event: Optional[Callable[[dict], None]] = None) -> None:
        self.steps: List[dict] = []
        self.llm_calls: List[dict] = []
        self.events: List[dict] = []
        self.on_event = on_event

    def _emit(self, event: dict) -> None:
        self.events.append(event)
        if self.on_event:
            self.on_event(event)

    def add_step(
        self,
        name: str,
        start: float,
        end: float,
        metadata: Optional[dict] = None,
    ) -> None:
        event = {
            "name": name,
            "start": start,
            "end": end,
            "duration": end - start,
            "metadata": metadata or {},
        }
        self.steps.append(event)
        self._emit({"type": "step", **event})

    def add_llm_call(
        self,
        stage: str,
        start: float,
        end: float,
        usage: Optional[dict] = None,
        chunk_index: Optional[int] = None,
    ) -> None:
        entry = {
            "stage": stage,
            "start": start,
            "end": end,
            "duration": end - start,
        }
        if chunk_index is not None:
            entry["chunk_index"] = chunk_index
        if usage:
            entry["usage"] = usage
        self.llm_calls.append(entry)
        self._emit({"type": "llm_call", **entry})


def _build_graph(
    llm: ChatOpenAI,
    chunk_size: int,
    chunk_overlap: int,
    usage_callback: Optional[UsageMetadataCallbackHandler],
    trace: Optional[TraceCollector],
):
    """Create and compile the LangGraph summarization workflow."""
    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You summarize legal documents. Produce concise bullet points that capture "
                    "key obligations, rights, restrictions, term/termination, fees, IP, liability, "
                    "and warranties where present. Keep a precise legal tone. Output only bullets."
                ),
            ),
            ("human", "Text:\n\n{text}"),
        ]
    )
    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Combine partial bullet summaries into a single coherent bullet list. "
                    "Remove duplicates, preserve critical legal details, and choose the number "
                    "of bullets that best fits the document. Output only bullets."
                ),
            ),
            ("human", "Partial summaries:\n\n{summaries}"),
        ]
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    def invoke_llm(messages, stage: str, chunk_index: Optional[int] = None):
        """Invoke the LLM and optionally capture usage metadata."""
        start_time = time.perf_counter() if trace else None
        if usage_callback is None:
            response = llm.invoke(messages)
        else:
            response = llm.invoke(messages, config={"callbacks": [usage_callback]})
        if trace and start_time is not None:
            usage = getattr(response, "usage_metadata", None) or {}
            trace.add_llm_call(
                stage=stage,
                start=start_time,
                end=time.perf_counter(),
                usage=usage,
                chunk_index=chunk_index,
            )
        return response

    def load_docs(state: SummaryState) -> SummaryState:
        """Load .docx or .pdf documents from the given file path."""
        step_start = time.perf_counter() if trace else None
        path = Path(state["file_path"]).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext == ".docx":
            loader = Docx2txtLoader(str(path))
        elif ext == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            raise ValueError("Unsupported file type. Use .docx or .pdf.")

        docs = loader.load()
        if not docs:
            raise ValueError("No content loaded from the file.")
        if trace and step_start is not None:
            trace.add_step("load_docs", step_start, time.perf_counter(), {"documents": len(docs)})
        return {"documents": docs}

    def split_docs(state: SummaryState) -> SummaryState:
        """Split documents into character-based chunks for summarization."""
        step_start = time.perf_counter() if trace else None
        documents = state["documents"]
        full_text = "\n\n".join(doc.page_content for doc in documents)
        splits = splitter.create_documents([full_text])
        if trace and step_start is not None:
            trace.add_step(
                "split_docs",
                step_start,
                time.perf_counter(),
                {"chunks": len(splits), "chars": len(full_text)},
            )
        return {"splits": splits}

    def map_summarize(state: SummaryState) -> SummaryState:
        """Summarize each chunk independently into bullet points."""
        step_start = time.perf_counter() if trace else None
        summaries: List[str] = []
        for index, doc in enumerate(state["splits"], start=1):
            messages = map_prompt.format_messages(text=doc.page_content)
            response = invoke_llm(messages, stage="map_summarize", chunk_index=index)
            summaries.append(response.content.strip())
        if trace and step_start is not None:
            trace.add_step(
                "map_summarize",
                step_start,
                time.perf_counter(),
                {"chunks": len(summaries)},
            )
        return {"chunk_summaries": summaries}

    def reduce_summaries(state: SummaryState) -> SummaryState:
        """Merge chunk summaries into a final bullet summary."""
        step_start = time.perf_counter() if trace else None
        summaries = state.get("chunk_summaries", [])
        if not summaries:
            if trace and step_start is not None:
                trace.add_step(
                    "reduce_summaries",
                    step_start,
                    time.perf_counter(),
                    {"summaries": 0},
                )
            return {"final_summary": ""}
        if len(summaries) == 1:
            if trace and step_start is not None:
                trace.add_step(
                    "reduce_summaries",
                    step_start,
                    time.perf_counter(),
                    {"summaries": 1},
                )
            return {"final_summary": summaries[0]}

        combined = "\n\n".join(summaries)
        messages = reduce_prompt.format_messages(summaries=combined)
        response = invoke_llm(messages, stage="reduce_summaries")
        final = response.content.strip()
        if trace and step_start is not None:
            trace.add_step(
                "reduce_summaries",
                step_start,
                time.perf_counter(),
                {"summaries": len(summaries)},
            )
        return {"final_summary": final}

    graph = StateGraph(SummaryState)
    graph.add_node("load_docs", load_docs)
    graph.add_node("split_docs", split_docs)
    graph.add_node("map_summarize", map_summarize)
    graph.add_node("reduce_summaries", reduce_summaries)
    graph.set_entry_point("load_docs")
    graph.add_edge("load_docs", "split_docs")
    graph.add_edge("split_docs", "map_summarize")
    graph.add_edge("map_summarize", "reduce_summaries")
    graph.add_edge("reduce_summaries", END)
    return graph.compile()


def run_summary(
    file_path: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    chunk_size: int = 25000,
    chunk_overlap: int = 400,
    show_raw_text: bool = False,
    show_chunks: bool = False,
    metrics: bool = False,
    graph_path: Optional[str] = None,
    trace: Optional[TraceCollector] = None,
) -> dict:
    """Run the summarization workflow and return outputs for UI or CLI."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env.")

    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    usage_callback = UsageMetadataCallbackHandler() if metrics else None
    graph = _build_graph(llm, chunk_size, chunk_overlap, usage_callback, trace)

    graph_error = None
    if graph_path:
        try:
            png_bytes = graph.get_graph().draw_mermaid_png()
            with open(graph_path, "wb") as handle:
                handle.write(png_bytes)
        except Exception as exc:
            graph_error = str(exc)

    start_time = time.perf_counter()
    result = graph.invoke({"file_path": file_path})
    elapsed = time.perf_counter() - start_time
    summary = result.get("final_summary", "").strip()

    if not summary:
        raise RuntimeError("No summary generated.")

    output = {"summary": summary}
    if show_raw_text:
        output["documents"] = result.get("documents", [])
    if show_chunks:
        output["chunk_summaries"] = result.get("chunk_summaries", [])
    if metrics:
        output["metrics"] = {
            "elapsed": elapsed,
            "usage": usage_callback.usage_metadata if usage_callback else {},
        }
    if graph_path:
        output["graph_path"] = graph_path
        if graph_error:
            output["graph_error"] = graph_error
    if trace:
        output["trace"] = trace
    return output


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments and return the populated namespace."""
    parser = argparse.ArgumentParser(
        description="Summarize a .docx or .pdf file with LangChain + LangGraph."
    )
    parser.add_argument("file", nargs="?", help="Path to a .docx or .pdf file")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name (default: gpt-4o-mini or OPENAI_MODEL).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the model (default: 0.2).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25000,
        help="Chunk size for splitting documents (default: 25000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=400,
        help="Chunk overlap for splitting documents (default: 400).",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print raw chunk summaries before the final summary.",
    )
    parser.add_argument(
        "--raw-text",
        action="store_true",
        help="Print the extracted text before summarization.",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print elapsed time and token usage statistics.",
    )
    parser.add_argument(
        "--graph",
        metavar="PATH",
        help="Write a PNG visualization of the LangGraph to the given path.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point; loads env, runs the graph, prints the summary."""
    load_dotenv()
    args = _parse_args()
    file_path = args.file or input("Enter file path (.docx or .pdf): ").strip()
    if not file_path:
        print("No file path provided.", file=sys.stderr)
        return 1

    try:
        result = run_summary(
            file_path=file_path,
            model=args.model,
            temperature=args.temperature,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            show_raw_text=args.raw_text,
            show_chunks=args.show_chunks,
            metrics=args.metrics,
            graph_path=args.graph,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.graph:
        if "graph_error" in result:
            print(f"Failed to render graph: {result['graph_error']}", file=sys.stderr)
        else:
            print(f"Wrote graph image to {result['graph_path']}")

    if args.raw_text:
        documents = result.get("documents", [])
        print("Raw extracted text:")
        if not documents:
            print("(no extracted text available)")
        for index, doc in enumerate(documents, start=1):
            print(f"\n--- Document {index} ---\n{doc.page_content}")
        print("\nFinal summary:")

    if args.show_chunks:
        chunk_summaries = result.get("chunk_summaries", [])
        print("Raw chunk summaries:")
        for index, text in enumerate(chunk_summaries, start=1):
            print(f"\n--- Chunk {index} ---\n{text}")
        print("\nFinal summary:")

    print(result["summary"])

    if args.metrics:
        print("\nMetrics:")
        metrics = result.get("metrics", {})
        print(f"Elapsed: {metrics.get('elapsed', 0.0):.2f}s")
        usage = metrics.get("usage", {})
        if not usage:
            print("Token usage: not reported by the provider.")
        else:
            for model_name, data in usage.items():
                input_tokens = data.get("input_tokens", 0)
                output_tokens = data.get("output_tokens", 0)
                total_tokens = data.get("total_tokens", 0)
                print(
                    f"{model_name}: input={input_tokens} output={output_tokens} total={total_tokens}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
