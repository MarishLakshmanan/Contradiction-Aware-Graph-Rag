import asyncio
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

from clients.embed import EmbeddingClient
from clients.semantic_scholar import SemanticScholarClient
from clients.store import VectorStore
from vectordb.chunking import ChunkMarkdown
from vectordb.fetch import FetchPDFs
from vectordb.pipeline import DocumentPipeline

load_dotenv()

EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]
EMBEDDING_MODEL_URL = os.environ["EMBEDDING_MODEL_URL"]
CHAT_MODEL_ID = os.environ["CHAT_MODEL_ID"]
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None

model = init_chat_model(model=CHAT_MODEL_ID)

QUERY_GEN_PROMPT = """You are turning a research topic into a search query for the arXiv API. arXiv search matches on title/abstract keywords, so the query should be a concise, keyword-focused phrase -- not a full sentence or question.

Examples:
Topic: "Do larger language models actually reason better, or just pattern-match better?"
Query: large language model reasoning versus pattern matching

Topic: "How effective is retrieval-augmented generation at reducing hallucinations in LLMs?"
Query: retrieval-augmented generation hallucination reduction large language models

Return ONLY the query text, with no preamble, labels, or quotes."""


def generate_search_query(topic: str) -> str:
    """
    Turns a research topic into a better arXiv search query using an LLM.
    """
    response = model.invoke(
        [SystemMessage(content=QUERY_GEN_PROMPT), HumanMessage(content=topic)]
    )
    return str(response.content).strip()


async def run_pipeline(
    topic: str,
    collection_name: str,
    num_of_pdfs: int = 50,
    concurrency: int = 8,
) -> None:
    """
    Fetches papers on `topic`, then downloads, converts, chunks, embeds, and
    stores each one -- all in-memory, no on-disk checkpoints.
    """
    query = generate_search_query(topic)
    fetcher = FetchPDFs(topic=topic, num_of_pdfs=num_of_pdfs, query=query)
    metadata = fetcher.fetch()

    embedding_client = EmbeddingClient(
        model_id=EMBEDDING_MODEL_ID, model_url=EMBEDDING_MODEL_URL
    ).model
    papers_store = VectorStore(collection_name=f"{collection_name}_paper")
    chunks_store = VectorStore(collection_name=f"{collection_name}_chunks")

    pipeline = DocumentPipeline(
        embedding_client=embedding_client,
        chunker=ChunkMarkdown(),
        papers_store=papers_store,
        chunks_store=chunks_store,
        semantic_scholar_client=SemanticScholarClient(api_key=SEMANTIC_SCHOLAR_API_KEY),
        concurrency=concurrency,
    )
    await pipeline.run(metadata)


if __name__ == "__main__":
    default_topic = "Do larger language models actually reason better, or just pattern-match better?"
    collection_name = "LLM"
    asyncio.run(run_pipeline(default_topic, collection_name))
