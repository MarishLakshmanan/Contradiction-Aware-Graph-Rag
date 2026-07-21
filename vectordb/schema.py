from pydantic import BaseModel
from typing import TypedDict, TypeAlias


class MetaData(TypedDict):
    id: str
    title: str
    pdf_url: str
    abstract: str


class ChunkMetadata(TypedDict):
    article_id: str
    pdf_url: str
    chunk_index: int


class PaperMetadata(TypedDict):
    title: str
    pdf_url: str
    citation_graph: str
    semantic_scholar_id: str


class EvalMetadata(TypedDict):
    article_id: str
    pdf_url: str
    chunk: str


ExtractedContent: TypeAlias = list[tuple[str, str]]
