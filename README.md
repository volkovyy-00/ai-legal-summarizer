# AI Legal Summarizer (Demo)

Summarize .docx or .pdf files with LangChain + LangGraph and OpenAI.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`) and set:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=ai-legal-demo
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

## Run

```bash
python summarize.py "Software License Agreement.docx"
```

Or run without arguments to enter the file path interactively.

## LangSmith tracing

LangChain picks up LangSmith configuration from environment variables. When
`LANGCHAIN_TRACING_V2=true`, each run will send a trace to your LangSmith project. Set `LANGCHAIN_TRACING_V2=false` if you do not need see tracing.

Optional flags:

```bash
python summarize.py "Synthetic_SaaS_Software_License_Agreement_CZ_Law_FINAL.pdf" \
  --raw-text --show-chunks --metrics
```

- `--raw-text` prints the extracted text before summarization.
- `--show-chunks` prints raw chunk summaries before the final summary.
- `--metrics` prints elapsed time and token usage stats.
- `--graph PATH` writes a PNG visualization of the workflow to the given path.

## Streamlit UI

```bash
streamlit run app.py
```

Upload a file and click **Summarize**. The workflow trace is available in the
collapsed **Trace** section and streams updates during the run.
