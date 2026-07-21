from pydantic import BaseModel
from typing import Literal, TypedDict, TypeAlias


class Chunk(TypedDict):
    text: str
    pages: set


class MetaData(TypedDict):
    id: str
    title: str
    pdf_url: str
    abstract: str


class ChunkMetadata(TypedDict):
    article_id: str
    pdf_url: str
    page_no: list


class PaperMetadata(TypedDict):
    title: str
    pdf_url: str
    citation_graph: str
    semantic_scholar_id: str


class EvalMetadata(TypedDict):
    article_id: str
    pdf_url: str
    chunk: str


class TopicMetadata(TypedDict):
    created_at: str
    num_of_pdfs: int
    max_chars: int
    chunk_overlap: int


class TopicMatch(TypedDict):
    collection_name: str
    topic: str
    distance: float


class IngestionOutcome(TypedDict):
    status: Literal["ingested", "conflict", "skipped_similar", "failed"]
    collection_name: str
    topic: str
    conflicting_topic: str | None
    match: TopicMatch | None
    error: str | None


ExtractedContent: TypeAlias = list[tuple[str, str]]
