import asyncio
import xml.etree.ElementTree as etree

import httpx
from httpx import AsyncClient
from langchain_openai import OpenAIEmbeddings

from clients.semantic_scholar import SemanticScholarClient
from clients.store import VectorStore
from vectordb.chunking import ChunkMarkdown
from vectordb.citation_graph import build_citation_graph, serialize_graph
from vectordb.converter import Converter
from vectordb.schema import ChunkMetadata, MetaData, PaperMetadata

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"


class DocumentPipeline:
    """
    Fuses download -> GROBID convert -> chunk -> embed -> store into a single
    async pipeline per paper, with nothing persisted to disk in between.
    A paper's metadata and its chunks are only written once that paper's own
    pipeline succeeds, independently of every other paper in the run.
    """

    def __init__(
        self,
        embedding_client: OpenAIEmbeddings,
        chunker: ChunkMarkdown,
        papers_store: VectorStore,
        chunks_store: VectorStore,
        semantic_scholar_client: SemanticScholarClient,
        concurrency: int = 8,
        max_attempts: int = 2,
    ):
        self.embedding_client = embedding_client
        self.chunker = chunker
        self.papers_store = papers_store
        self.chunks_store = chunks_store
        self.semantic_scholar_client = semantic_scholar_client
        self.semaphore = asyncio.Semaphore(concurrency)
        self.max_attempts = max_attempts
        # Semantic Scholar's public endpoint is limited to ~1 req/sec, independent
        # of the wider per-paper concurrency limit above
        self.s2_semaphore = asyncio.Semaphore(1)

    def __is_grobid_reachable(self) -> bool:
        try:
            httpx.head(GROBID_URL, timeout=5)
            return True
        except httpx.TransportError:
            return False

    async def __download_pdf(self, metadata: MetaData, client: AsyncClient) -> bytes:
        response = await client.get(url=metadata["pdf_url"], timeout=30)
        response.raise_for_status()
        return response.content

    async def __convert_to_markdown(self, pdf_bytes: bytes, client: AsyncClient) -> str:
        response = await client.post(GROBID_URL, files={"input": pdf_bytes}, timeout=60)
        response.raise_for_status()
        tree = etree.ElementTree(etree.fromstring(response.content))
        return Converter(tree).extract_content()

    async def __store_paper_and_chunks(
        self, metadata: MetaData, content: str, client: AsyncClient
    ) -> None:
        chunks = self.chunker.chunk(content)
        if not chunks:
            print(f"No chunks extracted for {metadata['id']}, skipping storage")
            return

        paper_text = f"{metadata['title']}\n\n{metadata['abstract']}"
        vectors = await self.embedding_client.aembed_documents(chunks + [paper_text])
        chunk_vectors, paper_vector = vectors[:-1], vectors[-1]

        # built before either store write below, so a hard failure here is safe
        # to retry the whole per-paper chain without risking a duplicate-id write
        async with self.s2_semaphore:
            graph, semantic_scholar_id = await build_citation_graph(
                self.semantic_scholar_client, client, metadata
            )
            # paces requests to Semantic Scholar's public endpoint to ~1/sec,
            # since its rate limit is per-time-window, not just per-concurrency
            await asyncio.sleep(2)
        citation_graph_json = serialize_graph(graph)

        ids = [f"{metadata['id']}-{i}" for i in range(len(chunks))]
        chunk_metadatas: list[ChunkMetadata] = [
            {
                "article_id": metadata["id"],
                "pdf_url": metadata["pdf_url"],
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        # both writes only happen once every prior step for this paper has succeeded,
        # so a paper's metadata and its chunks are always stored (or not stored) together
        await asyncio.to_thread(
            self.chunks_store.add,
            ids=ids,
            documents=chunks,
            embeddings=chunk_vectors,
            metadatas=chunk_metadatas,
        )
        paper_metadata: PaperMetadata = {
            "title": metadata["title"],
            "pdf_url": metadata["pdf_url"],
            "citation_graph": citation_graph_json,
            "semantic_scholar_id": semantic_scholar_id,
        }
        await asyncio.to_thread(
            self.papers_store.add,
            ids=[metadata["id"]],
            documents=[paper_text],
            embeddings=[paper_vector],
            metadatas=[paper_metadata],
        )

    async def __process_one(self, metadata: MetaData, client: AsyncClient) -> None:
        async with self.semaphore:
            for attempt in range(1, self.max_attempts + 1):
                try:
                    pdf_bytes = await self.__download_pdf(metadata, client)
                    content = await self.__convert_to_markdown(pdf_bytes, client)
                    await self.__store_paper_and_chunks(metadata, content, client)
                    return
                except Exception as exc:
                    if attempt == self.max_attempts:
                        raise
                    print(f"Retrying {metadata['id']} after error: {exc}")

    async def run(self, metadata: list[MetaData]) -> None:
        if not self.__is_grobid_reachable():
            raise Exception("Grobid isn't reachable")

        limits = httpx.Limits(
            max_connections=8, max_keepalive_connections=8, keepalive_expiry=5
        )
        async with httpx.AsyncClient(limits=limits) as client:
            results = await asyncio.gather(
                *[self.__process_one(m, client) for m in metadata],
                return_exceptions=True,
            )
            for m, res in zip(metadata, results):
                if isinstance(res, Exception):
                    print(f"Failed {m['id']}: {res}")
                else:
                    print(f"Stored {m['id']}")
