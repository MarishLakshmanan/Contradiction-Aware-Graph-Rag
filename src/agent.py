from dataclasses import dataclass
from functools import lru_cache
from typing import cast

from chromadb import Collection
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from src.clients.store import VectorStore
from settings import settings
from src.utils.citation_graph import deserialize_graph
from src.schema import ChunkMetadata, PaperMetadata


class Chunk(BaseModel):
    """A retrieved chunk with the source metadata needed to cite it."""

    text: str
    article_id: str
    pdf_url: str
    page_no: list


class RelatedPaper(BaseModel):
    """A paper connected via the citation graph, used as background context only."""

    article_id: str
    pdf_url: str
    content: str


class State(BaseModel):
    """Graph state threaded through rewrite_query -> retrieve_chunks -> expand_related_papers -> generate_answer."""

    query: str
    search_query: str = ""
    chunks: list[Chunk] = Field(default_factory=list)
    related_papers: list[RelatedPaper] = Field(default_factory=list)
    answer: str = ""


@dataclass
class AgentContext:
    """Per-invocation routing context: which topic's collections to query."""

    collection_name: str


@lru_cache(maxsize=None)
def _get_chunks_store(collection_name: str) -> Collection:
    return VectorStore(collection_name=f"{collection_name}_chunks").get()


@lru_cache(maxsize=None)
def _get_papers_store(collection_name: str) -> Collection:
    return VectorStore(collection_name=f"{collection_name}_papers").get()


model = init_chat_model(model=settings.chat_model_id)


REWRITE_PROMPT = """You are rewriting a user's question into a search query optimized for retrieving relevant passages from a vector database of research paper chunks.

Rewrite the query to be specific, keyword-rich, and stripped of conversational phrasing. Preserve the original meaning and intent exactly.

Examples:
User query: "does making a model bigger actually help it reason or is it just memorizing patterns?"
Rewritten: scaling model size effect on reasoning ability versus pattern matching

User query: "what's chain of thought and does it actually mean the model is reasoning"
Rewritten: chain-of-thought prompting reasoning versus pattern matching in large language models

User query: "are benchmarks like GSM8K actually measuring real reasoning"
Rewritten: GSM8K benchmark reasoning evaluation validity in large language models

Return ONLY the rewritten query text, with no preamble, labels, or quotes."""


ANSWER_PROMPT = """You are answering a user's question using only the provided context chunks from research papers.

Rules:
- Base your answer strictly on the context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so explicitly.
- Cite which article each fact comes from using its pdf_url.
- The "Context" section below contains directly retrieved excerpts -- this is your primary evidence for specific claims.
- The "Related papers" section contains only title+abstract-level background for papers connected via the citation graph to your primary sources -- use it only for broader background/framing, not as evidence for a specific granular claim, since it lacks excerpt-level detail. Still cite these using their pdf_url exactly as with primary context.

Example:
Context:
[article_id: http://arxiv.org/abs/2307.08072v2, pdf_url: https://arxiv.org/pdf/2307.08072v2, page_no: [4]]
Models above 60B parameters showed a marked increase in accuracy on multi-step arithmetic tasks, but this gain vanished when the surface form of the problem was perturbed, suggesting reliance on memorized patterns rather than robust reasoning.

Question: Do bigger models reason more robustly or just pattern-match?
Answer: According to https://arxiv.org/pdf/2307.08072v2, larger models (60B+ parameters) perform better on arithmetic tasks but lose that advantage when the problem's surface form is perturbed, indicating the gains come more from pattern matching than from robust reasoning.

Now answer the user's actual question using the context below.

Context:
{context}

Related papers (from the citation graph):
{related_context}"""


def rewrite_query(state: State) -> State:
    """
    Rewrites the user's conversational query into a keyword-rich search query.

    Args:
        state (State): graph state; reads state.query.

    Returns:
        State: state with state.search_query populated.
    """
    response = model.invoke(
        [SystemMessage(content=REWRITE_PROMPT), HumanMessage(content=state.query)]
    )
    state.search_query = str(response.content).strip()
    return state


def retrieve_chunks(state: State, runtime: Runtime[AgentContext]) -> State:
    """
    Queries the chunks collection for the top matches to the rewritten search query.

    Args:
        state (State): graph state; reads state.search_query.
        runtime (Runtime[AgentContext]): per-invocation context; reads collection_name.

    Returns:
        State: state with state.chunks populated.
    """
    chunks_store = _get_chunks_store(runtime.context.collection_name)
    response = chunks_store.query(
        query_texts=[state.search_query],
        n_results=5,
        include=["documents", "metadatas"],
    )
    documents = response["documents"][0] if response["documents"] else []
    metadatas = response["metadatas"][0] if response["metadatas"] else []
    state.chunks = [
        Chunk(
            text=document,
            article_id=cast(ChunkMetadata, metadata)["article_id"],
            pdf_url=cast(ChunkMetadata, metadata)["pdf_url"],
            page_no=cast(ChunkMetadata, metadata)["page_no"],
        )
        for document, metadata in zip(documents, metadatas)
    ]
    return state


def expand_related_papers(state: State, runtime: Runtime[AgentContext]) -> State:
    """
    Expands the retrieved chunks' source papers into their one-hop citation-graph neighbors.

    Args:
        state (State): graph state; reads state.chunks.
        runtime (Runtime[AgentContext]): per-invocation context; reads collection_name.

    Returns:
        State: state with state.related_papers populated.
    """
    papers_store = _get_papers_store(runtime.context.collection_name)

    # papers whose chunks were already retrieved -- used below to avoid re-adding
    # one of them as a "related" paper too
    retrieved_article_ids = {chunk.article_id for chunk in state.chunks}
    if not retrieved_article_ids:
        state.related_papers = []
        return state

    # pull each source paper's one-hop citation graph and union every neighbor's
    # Semantic Scholar paperId; each graph's own center id is subtracted out so a
    # paper is never counted as a "neighbor" of itself
    source_rows = papers_store.get(
        ids=list(retrieved_article_ids), include=["metadatas"]
    )
    candidate_s2_ids: set[str] = set()
    for metadata in source_rows["metadatas"] or []:
        meta = cast(PaperMetadata, metadata)
        graph = deserialize_graph(meta["citation_graph"])
        candidate_s2_ids |= set(graph.nodes) - {meta["semantic_scholar_id"]}

    if not candidate_s2_ids:
        state.related_papers = []
        return state

    # citation-graph nodes are Semantic Scholar paperIds, not our arXiv-based row
    # ids, so filter on the semantic_scholar_id metadata field directly rather
    # than scanning every row ourselves
    related_rows = papers_store.get(
        where={"semantic_scholar_id": {"$in": list(candidate_s2_ids)}},
        include=["documents", "metadatas"],
    )
    state.related_papers = [
        RelatedPaper(
            article_id=row_id,
            pdf_url=cast(PaperMetadata, metadata)["pdf_url"],
            content=document,
        )
        for row_id, document, metadata in zip(
            related_rows["ids"],
            related_rows["documents"] or [],
            related_rows["metadatas"] or [],
        )
        # drop a match that's already one of the directly retrieved chunks' own source papers
        if row_id not in retrieved_article_ids
    ]
    return state


def generate_answer(state: State) -> State:
    """
    Answers the original query from the retrieved chunks and related-paper context.

    Args:
        state (State): graph state; reads state.query, state.chunks, state.related_papers.

    Returns:
        State: state with state.answer populated.
    """
    context = "\n\n".join(
        f"[article_id: {chunk.article_id}, pdf_url: {chunk.pdf_url}, chunk_index: {chunk.page_no}]\n{chunk.text}"
        for chunk in state.chunks
    )
    related_context = (
        "\n\n".join(
            f"[article_id: {paper.article_id}, pdf_url: {paper.pdf_url}]\n{paper.content}"
            for paper in state.related_papers
        )
        or "(none found)"
    )
    response = model.invoke(
        [
            SystemMessage(
                content=ANSWER_PROMPT.format(
                    context=context, related_context=related_context
                )
            ),
            HumanMessage(content=state.query),
        ]
    )
    state.answer = str(response.content)
    return state


builder = StateGraph(State, context_schema=AgentContext)

builder.add_node("rewrite_query", rewrite_query)
builder.add_node("retrieve_chunks", retrieve_chunks)
builder.add_node("expand_related_papers", expand_related_papers)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "rewrite_query")
builder.add_edge("rewrite_query", "retrieve_chunks")
builder.add_edge("retrieve_chunks", "expand_related_papers")
builder.add_edge("expand_related_papers", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()
