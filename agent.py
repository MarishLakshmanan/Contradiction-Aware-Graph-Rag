import os
from dotenv import load_dotenv

load_dotenv()

from typing import cast

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from clients.embed import EmbeddingClient
from clients.store import VectorStore
from vectordb.citation_graph import deserialize_graph
from vectordb.schema import ChunkMetadata, PaperMetadata

EMBEDDING_MODEL_ID, EMBEDDING_MODEL_URL = (
    os.environ["EMBEDDING_MODEL_ID"],
    os.environ["EMBEDDING_MODEL_URL"],
)

CHAT_MODEL_ID = os.environ["CHAT_MODEL_ID"]


class Chunk(BaseModel):
    text: str
    article_id: str
    pdf_url: str
    chunk_index: int


class RelatedPaper(BaseModel):
    article_id: str
    pdf_url: str
    content: str


class State(BaseModel):
    query: str
    search_query: str = ""
    chunks: list[Chunk] = Field(default_factory=list)
    related_papers: list[RelatedPaper] = Field(default_factory=list)
    answer: str = ""


model = init_chat_model(model=CHAT_MODEL_ID)
embedding_client = EmbeddingClient(
    model_id=EMBEDDING_MODEL_ID, model_url=EMBEDDING_MODEL_URL
).model
chunks_store = VectorStore(collection_name="LLM_chunks").get()
papers_store = VectorStore(collection_name="LLM_paper").get()


REWRITE_PROMPT = """You are rewriting a user's question into a search query optimized for retrieving relevant passages from a vector database of research paper chunks. The papers all address one fixed topic: whether larger language models actually reason better, or just pattern-match better.

Rewrite the query to be specific, keyword-rich, and stripped of conversational phrasing. Preserve the original meaning and intent exactly.

Examples:
User query: "does making a model bigger actually help it reason or is it just memorizing patterns?"
Rewritten: scaling model size effect on reasoning ability versus pattern matching

User query: "what's chain of thought and does it actually mean the model is reasoning"
Rewritten: chain-of-thought prompting reasoning versus pattern matching in large language models

User query: "are benchmarks like GSM8K actually measuring real reasoning"
Rewritten: GSM8K benchmark reasoning evaluation validity in large language models

Return ONLY the rewritten query text, with no preamble, labels, or quotes."""


ANSWER_PROMPT = """You are answering a user's question using only the provided context chunks from research papers on whether larger language models actually reason better, or just pattern-match better.

Rules:
- Base your answer strictly on the context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so explicitly.
- Cite which article each fact comes from using its pdf_url.
- The "Context" section below contains directly retrieved excerpts -- this is your primary evidence for specific claims.
- The "Related papers" section contains only title+abstract-level background for papers connected via the citation graph to your primary sources -- use it only for broader background/framing, not as evidence for a specific granular claim, since it lacks excerpt-level detail. Still cite these using their pdf_url exactly as with primary context.

Example:
Context:
[article_id: http://arxiv.org/abs/2307.08072v2, pdf_url: https://arxiv.org/pdf/2307.08072v2, chunk_index: 4]
Models above 60B parameters showed a marked increase in accuracy on multi-step arithmetic tasks, but this gain vanished when the surface form of the problem was perturbed, suggesting reliance on memorized patterns rather than robust reasoning.

Question: Do bigger models reason more robustly or just pattern-match?
Answer: According to https://arxiv.org/pdf/2307.08072v2, larger models (60B+ parameters) perform better on arithmetic tasks but lose that advantage when the problem's surface form is perturbed, indicating the gains come more from pattern matching than from robust reasoning.

Now answer the user's actual question using the context below.

Context:
{context}

Related papers (from the citation graph):
{related_context}"""


def rewrite_query(state: State) -> State:
    response = model.invoke(
        [SystemMessage(content=REWRITE_PROMPT), HumanMessage(content=state.query)]
    )
    state.search_query = str(response.content).strip()
    return state


def retrieve_chunks(state: State) -> State:
    embedding = embedding_client.embed_query(state.search_query)
    response = chunks_store.query(
        query_embeddings=[embedding],
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
            chunk_index=cast(ChunkMetadata, metadata)["chunk_index"],
        )
        for document, metadata in zip(documents, metadatas)
    ]
    return state


def expand_related_papers(state: State) -> State:
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
    context = "\n\n".join(
        f"[article_id: {chunk.article_id}, pdf_url: {chunk.pdf_url}, chunk_index: {chunk.chunk_index}]\n{chunk.text}"
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


builder = StateGraph(State)

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
