import asyncio
import logging

from datetime import datetime, timezone

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

from src.schema import IngestionOutcome, TopicMetadata

from src.clients.semantic_scholar import SemanticScholarClient
from src.clients.store import VectorStore
from settings import settings
from src.utils.fetch import FetchPDFs
from src.pipeline import DocumentPipeline
from src.utils.topics import (
    TOPICS_COLLECTION_NAME,
    find_collection_name,
    find_similar_topic,
    register_topic,
)

logger = logging.getLogger(__name__)

model = init_chat_model(model=settings.chat_model_id)

# L2 distance (Chroma's default space, not cosine similarity -- see
# src/utils/topics.py) below which two topics are treated as duplicates.
# Calibrated once against a reworded-duplicate topic (~0.27) vs. unrelated
# topics (~1.3-1.4) -- see tests/topics.py. Revisit if real usage disagrees.
TOPIC_SIMILARITY_THRESHOLD = 0.4

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

    Args:
        topic (str): research topic/question to convert into a search query.

    Returns:
        str: concise, keyword-focused arXiv search query.
    """
    response = model.invoke(
        [SystemMessage(content=QUERY_GEN_PROMPT), HumanMessage(content=topic)]
    )
    query = str(response.content).strip()
    logger.info("Generated arXiv search query: %r", query)
    return query


async def run_pipeline(
    topic: str,
    collection_name: str,
    max_chars: int,
    chunk_overlap: int,
    num_of_pdfs: int = 50,
    concurrency: int = 8,
) -> None:
    """
    Fetches papers on `topic`, then downloads, converts, chunks, embeds, and
    stores each one -- all in-memory, no on-disk checkpoints.

    Args:
        topic (str): research topic to fetch and process papers for.
        collection_name (str): base Chroma collection name; `_paper`/`_chunks` are appended.
         max_chars: Max number of character allowed in a chunk
        chunk_overlap: Character allowed to overlap
        num_of_pdfs (int): number of reachable PDFs to fetch.
        concurrency (int): max papers processed concurrently.
    """
    logger.info(
        "Starting pipeline: topic=%r collection_name=%r num_of_pdfs=%d",
        topic,
        collection_name,
        num_of_pdfs,
    )
    query = generate_search_query(topic)
    fetcher = FetchPDFs(topic=topic, num_of_pdfs=num_of_pdfs, query=query)
    metadata = fetcher.fetch()
    logger.info("Fetch complete: %d papers ready for processing", len(metadata))

    papers_store = VectorStore(
        collection_name=f"{collection_name}_papers",
    )
    chunks_store = VectorStore(
        collection_name=f"{collection_name}_chunks",
    )

    pipeline = DocumentPipeline(
        papers_store=papers_store,
        chunks_store=chunks_store,
        max_chunking_chars=max_chars,
        chunking_overlap=chunk_overlap,
        semantic_scholar_client=SemanticScholarClient(
            api_key=settings.semantic_scholar_api_key
        ),
        concurrency=concurrency,
    )
    await pipeline.run(metadata)
    logger.info("Pipeline run complete")


async def start_topic_ingestion(
    topic: str,
    collection_name: str,
    max_chars: int,
    chunk_overlap: int,
    num_of_pdfs: int = 50,
    concurrency: int = 8,
    force: bool = False,
) -> IngestionOutcome:
    """
    Checks the topics registry before running the pipeline: refuses to reuse a
    `collection_name` already registered to a different topic, and (unless
    `force`) skips ingestion if a near-duplicate topic is already registered.
    Registers the topic only after `run_pipeline` succeeds, so a failed run
    never leaves a registry entry pointing at empty collections.

    Args:
        topic (str): research topic to fetch and process papers for.
        collection_name (str): base Chroma collection name for this topic.
        max_chars: Max number of character allowed in a chunk
        chunk_overlap: Character allowed to overlap
        num_of_pdfs (int): number of reachable PDFs to fetch.
        concurrency (int): max papers processed concurrently.
        force (bool): if True, ingest even if a near-duplicate topic exists.

    Returns:
        IngestionOutcome: which of ingested/conflict/skipped_similar/failed happened.
    """
    registry = VectorStore(collection_name=TOPICS_COLLECTION_NAME)

    existing_topic = find_collection_name(collection_name, registry)
    if existing_topic is not None and existing_topic != topic:
        logger.error(
            "collection_name %r is already registered to a different topic (%r); "
            "refusing to reuse it for %r",
            collection_name,
            existing_topic,
            topic,
        )
        return {
            "status": "conflict",
            "collection_name": collection_name,
            "topic": topic,
            "conflicting_topic": existing_topic,
            "match": None,
            "error": None,
        }

    if not force:
        match = find_similar_topic(topic, registry, TOPIC_SIMILARITY_THRESHOLD)
        if match is not None:
            logger.info(
                "A similar topic is already registered: collection_name=%r "
                "topic=%r (distance=%.4f) -- use that collection instead, or "
                "pass force=True to ingest anyway",
                match["collection_name"],
                match["topic"],
                match["distance"],
            )
            return {
                "status": "skipped_similar",
                "collection_name": collection_name,
                "topic": topic,
                "conflicting_topic": None,
                "match": match,
                "error": None,
            }

    try:
        await run_pipeline(
            topic, collection_name, max_chars, chunk_overlap, num_of_pdfs, concurrency
        )
    except Exception as exc:
        logger.error("Ingestion failed for topic %r: %s", topic, exc)
        return {
            "status": "failed",
            "collection_name": collection_name,
            "topic": topic,
            "conflicting_topic": None,
            "match": None,
            "error": str(exc),
        }

    topic_metadata: TopicMetadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_overlap": chunk_overlap,
        "max_chars": max_chars,
        "num_of_pdfs": num_of_pdfs,
    }

    register_topic(
        collection_name=collection_name,
        topic=topic,
        metadata=topic_metadata,
        registry=registry,
    )
    return {
        "status": "ingested",
        "collection_name": collection_name,
        "topic": topic,
        "conflicting_topic": None,
        "match": None,
        "error": None,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    default_topic = "Do larger language models actually reason better, or just pattern-match better?"
    collection_name = "llm"
    num_of_pdfs = 10
    max_chars = 1000
    chunk_overlap = 200

    if num_of_pdfs < 10 or num_of_pdfs > 50:
        raise Exception(
            "Number of pdfs should be greater then 10 and less than or equal to 50"
        )
    if max_chars < 1000 or max_chars > 10000:
        raise Exception("Max character should be less then 10000 and greater than 1000")
    if chunk_overlap > max_chars // 2:
        raise Exception("Chunk overlap can't be greater than half of Max character")
    asyncio.run(
        start_topic_ingestion(
            default_topic,
            collection_name,
            num_of_pdfs=10,
            max_chars=max_chars,
            chunk_overlap=chunk_overlap,
        )
    )
