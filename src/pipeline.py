import asyncio
import logging

import httpx
from httpx import AsyncClient

from src.clients.semantic_scholar import (
    SemanticScholarClient,
)
from src.clients.store import VectorStore
from settings import settings
from src.utils.chunking import chunk_pdf_blocks
from src.utils.citation_graph import build_citation_graph, serialize_graph
from src.schema import ChunkMetadata, MetaData, PaperMetadata, Chunk

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """
    Fuses download -> chunk -> embed -> store into a single
    async pipeline per paper, with nothing persisted to disk in between.
    A paper's metadata and its chunks are only written once that paper's own
    pipeline succeeds, independently of every other paper in the run.
    """

    def __init__(
        self,
        papers_store: VectorStore,
        chunks_store: VectorStore,
        semantic_scholar_client: SemanticScholarClient,
        max_chunking_chars: int,
        chunking_overlap: int,
        concurrency: int = 8,
        max_attempts: int = 2,
    ):
        """

        Args:
            papers_store (VectorStore): The collection to store the Papers metadata.
            chunks_store (VectorStore): The collection to store the chunks generated from the paper and its metadata.
            semantic_scholar_client (SemanticScholarClient): The Semantic Scholar client. Which could have been instantiated here in the __init__.
            max_chunking_chars (int, optional): The Max number of chars allowed in a chunk.
            chunking_overlap (int, optional): The allowed overlap between chunks. This value is within the max_chunking_chars value.
            concurrency (int, optional): Max number of tasks allowed to enter the coroutines . Defaults to 8.
            max_attempts (int, optional): Max number of attempts for a paper. if it fails. Defaults to 2.
        """

        self.papers_store = papers_store
        self.chunks_store = chunks_store
        self.semantic_scholar_client = semantic_scholar_client
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.max_attempts = max_attempts
        self.max_chunking_chars = max_chunking_chars
        self.chunking_overlap = chunking_overlap
        # Semantic Scholar's public endpoint is limited to ~1 req/sec, independent
        # of the wider per-paper concurrency limit above
        self.s2_semaphore = asyncio.Semaphore(1)

    async def __download_pdf(self, metadata: MetaData, client: AsyncClient) -> bytes:
        """
        Args:
            metadata (MetaData): paper metadata; reads metadata["pdf_url"].
            client (AsyncClient): shared async HTTP client.

        Returns:
            bytes: raw PDF content.
        """
        response = await client.get(url=metadata["pdf_url"], timeout=30)
        response.raise_for_status()
        return response.content

    async def __store_paper_and_chunks(
        self, metadata: MetaData, chunks: list[Chunk], client: AsyncClient
    ) -> None:
        """
        Prepare the Metadata for paper and chunks collection and add them to the respective collections
        Args:
            metadata (MetaData): The Metadata contains information about the paper being processed. It contains ids, and pdf_url
            chunks (list[Chunk]): The chunks generated from the PDF_url in the metadata.
            client (AsyncClient): The httpx async client for processing Semantic Scholar request
        """

        if not chunks:
            logger.warning(
                "No chunks extracted for %s, skipping storage", metadata["id"]
            )
            return
        logger.info("Chunked %s: %d chunks", metadata["id"], len(chunks))

        paper_text = f"{metadata['title']}\n\n{metadata['abstract']}"

        # built before either store write below, so a hard failure here is safe
        # to retry the whole per-paper chain without risking a duplicate-id write
        async with self.s2_semaphore:
            graph, semantic_scholar_id = await build_citation_graph(
                self.semantic_scholar_client, client, metadata
            )
            # paces requests to Semantic Scholar's public endpoint to ~1/sec,
            # since its rate limit is per-time-window, not just per-concurrency
            await asyncio.sleep(3)
        citation_graph_json = serialize_graph(graph)

        ids = [f"{metadata['id']}-{i}" for i in range(len(chunks))]
        documents = [chunk["text"] for chunk in chunks]
        page_nos = [list(chunk["pages"]) for chunk in chunks]

        chunk_metadatas: list[ChunkMetadata] = [
            {
                "article_id": metadata["id"],
                "pdf_url": metadata["pdf_url"],
                "page_no": page_no,
            }
            for page_no in page_nos
        ]
        # both writes only happen once every prior step for this paper has succeeded,
        # so a paper's metadata and its chunks are always stored (or not stored) together

        # Adding the chunk metadata
        await asyncio.to_thread(
            self.chunks_store.add,
            ids=ids,
            documents=documents,
            metadatas=chunk_metadatas,
        )
        paper_metadata: PaperMetadata = {
            "title": metadata["title"],
            "pdf_url": metadata["pdf_url"],
            "citation_graph": citation_graph_json,
            "semantic_scholar_id": semantic_scholar_id,
        }

        # Adding the Paper
        await asyncio.to_thread(
            self.papers_store.add,
            ids=[metadata["id"]],
            documents=[paper_text],
            metadatas=[paper_metadata],
        )
        logger.info(
            "Stored %s: %d chunks + paper metadata", metadata["id"], len(chunks)
        )

    async def __process_one(self, metadata: MetaData, client: AsyncClient) -> None:
        """
        Runs the download -> chunk -> store chain for one paper, retrying the
        whole chain up to max_attempts times before letting the error propagate.

        Args:
            metadata (MetaData): paper metadata to process.
            client (AsyncClient): shared async HTTP client.
        """
        async with self.semaphore:
            for attempt in range(1, self.max_attempts + 1):
                try:
                    pdf_bytes = await self.__download_pdf(metadata, client)
                    chunks = chunk_pdf_blocks(
                        pdf_bytes=pdf_bytes,
                        max_chars=self.max_chunking_chars,
                        overlap=self.chunking_overlap,
                    )
                    await self.__store_paper_and_chunks(metadata, chunks, client)
                    return

                except Exception:
                    if attempt == self.max_attempts:
                        logger.error(
                            "Giving up on %s after %d attempts",
                            metadata["id"],
                            attempt,
                            exc_info=True,
                        )
                        raise
                    logger.warning(
                        "Retrying %s after error (attempt %d/%d)",
                        metadata["id"],
                        attempt,
                        self.max_attempts,
                        exc_info=True,
                    )

    async def run(self, metadata: list[MetaData]) -> None:
        """
        Runs the per-paper pipeline over every paper concurrently, then logs a
        succeeded/failed summary. A failed paper doesn't abort the others.

        Args:
            metadata (list[MetaData]): papers to process.
        """
        logger.info(
            "Starting per-paper pipeline for %d papers (concurrency=%d)",
            len(metadata),
            self.concurrency,
        )

        limits = httpx.Limits(
            max_connections=8, max_keepalive_connections=8, keepalive_expiry=5
        )
        async with httpx.AsyncClient(limits=limits) as client:
            results = await asyncio.gather(
                *[self.__process_one(m, client) for m in metadata],
                return_exceptions=True,
            )
            failed = 0
            for m, res in zip(metadata, results):
                if isinstance(res, Exception):
                    failed += 1
                    logger.error("Failed %s: %s", m["id"], res, exc_info=res)
            logger.info(
                "Pipeline finished: %d succeeded, %d failed",
                len(metadata) - failed,
                failed,
            )
