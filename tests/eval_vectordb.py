import random
import xml.etree.ElementTree as etree
from typing import cast

import httpx
from langchain.chat_models import init_chat_model
from pydantic import SecretStr
from langchain.messages import HumanMessage, SystemMessage

from clients.embed import EmbeddingClient
from clients.store import VectorStore
from make import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_URL
from vectordb.converter import Converter
from vectordb.schema import ChunkMetadata, EvalMetadata
from dotenv import load_dotenv

load_dotenv()

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"


def fetch_documents(collection_name: str, n: int = 3) -> list[dict]:
    """
    Fetch n random entries from the given collection
    """
    store = VectorStore(collection_name=collection_name)
    all_ids = store.collection.get(include=[])["ids"]
    sample_ids = random.sample(all_ids, min(n, len(all_ids)))
    records = store.collection.get(ids=sample_ids, include=["documents", "metadatas"])
    assert records["documents"] is not None and records["metadatas"] is not None
    return [
        {"id": doc_id, "abstract": document, "pdf_url": metadata["pdf_url"]}
        for doc_id, document, metadata in zip(
            records["ids"], records["documents"], records["metadatas"]
        )
    ]


def convert_and_extract(pdf_url: str) -> str:
    """
    Downloads the pdf, convert it into xml using GROBID, then convert that into md
    """
    pdf_response = httpx.get(pdf_url, timeout=30)
    pdf_response.raise_for_status()
    grobid_response = httpx.post(
        GROBID_URL, files={"input": pdf_response.content}, timeout=60
    )
    grobid_response.raise_for_status()
    tree = etree.ElementTree(etree.fromstring(grobid_response.content))
    return Converter(tree).extract_content()


def invoke_llm(abstract: str, chunk: str) -> str:
    """
    Gives the abstract and chunk to the llm and get an query back
    """
    query_generate_prompt = query_generate_prompt = """
You are an expert in building and evaluating RAG (Retrieval-Augmented Generation) pipelines. You are helping me build a test dataset for evaluating a RAG pipeline.

You will be given an abstract of a document and a large chunk of text ("big_chunk") taken from that document.

Task:
1. Read the abstract to understand the overall context of the document.
2. Read the big_chunk carefully.
3. Generate ONE realistic user query that a person might type into a search or RAG system, such that the ideal retrieved answer would be a smaller, specific subset of the big_chunk (not the entire big_chunk).

Important context: the big_chunk provided here is intentionally larger than the chunks stored in the vector database. So the query you generate should target a specific fact, idea, or passage within the big_chunk — not the chunk as a whole — since retrieval will return a smaller subset of it.

Requirements for the query:
- It must be answerable using only information contained in the big_chunk.
- It should be specific enough that it maps to a sub-section of the big_chunk, not the entire thing.
- It should read like a natural question or search query a real user would ask (not a summary or restatement of the chunk).
- Do not reference "the text", "the document", "the chunk", or "the abstract" in the query — write it as a standalone question.

Output format:
- Return ONLY the query text.
- Do NOT include any preamble, explanation, labels, quotes, or phrases like "Here's the query:".
- Do NOT include markdown formatting.

<abstract>
{abstract}
</abstract>

<big_chunk>
{chunk}
</big_chunk>

Example:

abstract: "This paper talks about Earth's atmosphere"
big_chunk: "The sky appears blue because Earth's atmosphere scatters sunlight in all directions. White sunlight is made of all the colors of the rainbow, but blue light travels in shorter, smaller waves. These waves scatter much more than other colors when they hit gas molecules in the air."
output: Why is the sky blue?

Now generate the query for the abstract and big_chunk given above. Return only the query.
"""
    system_prompt = SystemMessage(
        content=query_generate_prompt.format(abstract=abstract, chunk=chunk)
    )

    model = init_chat_model(model="claude-haiku-4-5-20251001")
    response = model.invoke([system_prompt, HumanMessage(content="Generate the query")])
    print(response.content)
    return str(response.content)


def _random_big_chunk(content: str, size: int = 3000) -> str:
    if len(content) <= size:
        return content
    start = random.randint(0, len(content) - size)
    return content[start : start + size]


def build_eval_dataset(collection_name: str, n: int) -> None:
    """
    First fetch the documents
    then for each document convert it into md
    get a big chunk from the markdown and the abstract
    send both to the llm and the get the query
    now create an collection collection called paper_test and add the article_id, query, chunk
    """
    documents = fetch_documents(collection_name, n)
    embedding_client = EmbeddingClient(
        model_id=EMBEDDING_MODEL_ID, model_url=EMBEDDING_MODEL_URL
    ).model
    eval_store = VectorStore(collection_name=f"{collection_name}_test")
    eval_store.clear()

    for doc in documents:
        try:
            content = convert_and_extract(doc["pdf_url"])
            big_chunk = _random_big_chunk(content)
            if not big_chunk.strip():
                raise ValueError("converted markdown is empty")

            query = invoke_llm(doc["abstract"], big_chunk)
            if not query:
                raise ValueError("LLM returned an empty query")

            vector = embedding_client.embed_query(query)
            eval_metadata: EvalMetadata = {
                "article_id": doc["id"],
                "pdf_url": doc["pdf_url"],
                "chunk": big_chunk,
            }
            eval_store.add(
                ids=[doc["id"]],
                documents=[query],
                embeddings=[vector],
                metadatas=[eval_metadata],
            )
            print(f"Added eval entry for {doc['id']}")
        except Exception as exc:
            print(f"Skipping {doc['id']}: {exc}")


def evaluate_rag(
    collection_name: str, test_collection_name: str, recall: int = 5
) -> None:
    """
    This evaluates the build collection by using the test collection
    1. fetch all from the test collection
    2. use the query and fetch top 5 documents from the build collection
    3. check if the metadata from test_collection item is a superset of the content returned by documents collection
    """
    collection = VectorStore(collection_name=collection_name).get()
    test_collection = VectorStore(collection_name=test_collection_name).get()

    test_set = test_collection.get(include=["documents", "metadatas", "embeddings"])
    assert test_set["documents"] is not None
    assert test_set["metadatas"] is not None
    assert test_set["embeddings"] is not None

    chunk_hits = 0
    article_hits = 0
    total = 0
    for query, metadata, embedding in zip(
        test_set["documents"], test_set["metadatas"], test_set["embeddings"]
    ):
        total += 1
        response = collection.query(
            query_embeddings=[list(embedding)],
            n_results=recall,
            include=["documents", "metadatas"],
        )
        retrieved_chunks = response["documents"][0] if response["documents"] else []
        if not retrieved_chunks:
            print(f"{query}: no results retrieved")
            continue

        big_chunk = str(metadata.get("chunk", ""))
        hit_1 = any(
            retrieved_chunk in big_chunk for retrieved_chunk in retrieved_chunks
        )
        chunk_hits += hit_1

        retrieved_metadata = response["metadatas"][0] if response["metadatas"] else []
        test_article_id = cast(EvalMetadata, metadata)["article_id"]

        hit_2 = any(
            test_article_id == cast(ChunkMetadata, data)["article_id"]
            for data in retrieved_metadata
        )
        article_hits += hit_2
        print(
            f" {"Chunk HIT" if hit_1 else 'Chunk MISS'} , {'Article HIT' if hit_2 else 'Article MISS'}"
        )

    if total == 0:
        print("Test collection is empty -- run build_eval_dataset first")
        return
    print(f"\n{chunk_hits}/{total} chunk hits and {article_hits}/{total} article hits")


if __name__ == "__main__":
    build_eval_dataset("papers", 10)

    evaluate_rag("chunks", "papers_test", recall=5)
