"""Test for Vector Store"""

from src.clients.store import VectorStore
from chromadb import CloudClient
from chromadb.api import ClientAPI
from settings import settings

DOC_IDS = ["doc-1", "doc-2", "doc-3"]
DOCUMENTS = [
    "The sky appears blue because the atmosphere scatters short-wavelength light.",
    "Dogs are domesticated mammals commonly kept as household pets.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
]
METADATAS = [{"topic": "physics"}, {"topic": "biology"}, {"topic": "biology"}]


def test_collection_created(client: ClientAPI, collection_name: str) -> None:
    collections = [c.name for c in client.list_collections()]
    assert (
        collection_name in collections
    ), f"collection {collection_name!r} was not created"
    print(f"[create] OK - collection {collection_name!r} exists")


def test_get(store: VectorStore) -> None:
    store.add(ids=DOC_IDS, documents=DOCUMENTS, metadatas=METADATAS)

    result = store.collection.get(ids=DOC_IDS, include=["documents", "metadatas"])
    assert result["ids"] == DOC_IDS, f"expected ids {DOC_IDS}, got {result['ids']}"
    assert (
        result["documents"] == DOCUMENTS
    ), "stored documents don't match what was added"

    print(f"[get] OK - {len(DOC_IDS)} records stored and read back correctly")


def test_query(store: VectorStore) -> None:
    response = store.collection.query(
        query_texts=["Why does the atmosphere make the sky look blue?"],
        n_results=1,
        include=["documents"],
    )
    assert response["ids"] and response["ids"][0], "query returned no results"

    top_id = response["ids"][0][0]
    assert top_id == "doc-1", f"expected closest match doc-1, got {top_id}"

    print(f"[query] OK - closest match to the physics query was {top_id!r}")


if __name__ == "__main__":
    collection_name = "test"
    client = CloudClient(
        api_key=settings.chroma_api_key,
        tenant=settings.chroma_tenant_id,
        database=settings.chroma_db_name,
    )

    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)

    store = VectorStore(collection_name=collection_name)

    # Run all the tests
    test_collection_created(client, collection_name)
    test_get(store)
    test_query(store)

    client.delete_collection(collection_name)
