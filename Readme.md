# Graph RAG

Graph RAG lets you ingest research papers on any topic from arXiv, then ask questions against them through a LangGraph agent. Answers are grounded in the retrieved paper excerpts, and the agent also pulls in related papers from each source's citation graph for extra background. Topics aren't fixed: you can ingest as many as you want, and each chat picks which one to query.

## Technologies

- **Python 3.13**, managed with **uv** (`pyproject.toml` + `uv.lock`)
- **LangChain / LangGraph**: the agent graph, query rewriting, answer generation
- **Anthropic Claude**: the chat/reasoning model (`init_chat_model`)
- **HuggingFace** (router inference API): embeddings
- **Chroma Cloud**: vector storage (chunks, paper metadata, the topics registry)
- **PyMuPDF**: in-process PDF text extraction (no external conversion service)
- **arXiv API**: paper discovery/fetch
- **Semantic Scholar API** + **NetworkX**: one-hop citation graph per paper
- **Gradio**: the chat UI and the ingestion pipeline UI
- **LangGraph Platform (LangSmith)**: hosts the deployed agent server
- **Hugging Face Spaces**: hosts both deployed Gradio apps

## Pipeline

**Ingestion** turns a topic into stored, queryable data. Give it a topic and it fetches matching papers from arXiv, downloads and chunks each one's text, embeds the chunks, builds a citation graph for each paper via Semantic Scholar, and stores everything in Chroma Cloud. Once a topic's papers are all in, it's registered and ready to query.

**Inference** answers a question against one topic's ingested data. The agent rewrites your question into a better search query, retrieves the most relevant chunks, pulls in related papers from those chunks' citation graphs for extra context, and generates an answer grounded in the retrieved excerpts, citing sources.

## How the LangGraph server connects it all

The agent runs on a LangGraph server (locally via `langgraph dev`, or hosted on LangSmith). The chat UI talks to that server over the network for every question, telling it which topic to use, and streams the answer back as it's generated. The ingestion UI doesn't talk to the LangGraph server at all; it ingests data on its own.

Because the two are separate, switching between local dev and a real deployment is just a config change in `.env`, no code changes needed.

## Running locally

1. **Clone and install**
   ```
   uv sync
   ```
2. **Configure `.env`**: copy [.env.example](.env.example) to `.env` and fill in real values (Anthropic, Chroma Cloud, HuggingFace, optionally Semantic Scholar and LangSmith). For local dev, `LANGGRAPH_CLIENT_URL` should stay `http://localhost:2024`.
3. **Start everything at once:**
   ```
   uv run python -m src.dev
   ```
   This launches `langgraph dev`, the chat UI (http://localhost:7860), and the ingestion UI (http://localhost:7861) together, and stops all three cleanly on Ctrl+C.

   Or run each separately, in three terminals, if you want isolated logs:
   ```
   uv run langgraph dev
   uv run python -m src.UI.agent_ui
   uv run python -m src.UI.pipeline_ui
   ```
4. **Use it**: open the ingestion UI to ingest a new topic (or reuse the existing `llm` one), then open the chat UI, pick a topic, and ask a question.

See [CLAUDE.md](CLAUDE.md) for full implementation detail on every part of this pipeline.
