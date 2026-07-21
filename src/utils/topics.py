from typing import cast

from src.clients.store import VectorStore
from src.schema import TopicMatch, TopicMetadata

TOPICS_COLLECTION_NAME = "topics"


def find_similar_topic(
    topic: str, registry: VectorStore, threshold: float
) -> TopicMatch | None:
    """
    Looks for an existing topic whose embedding is within `threshold` (Chroma's
    default L2 space) of `topic`.

    Args:
        topic (str): topic text to check against the registry.
        registry (VectorStore): the topics registry collection.
        threshold (float): max L2 distance to count as a match.

    Returns:
        TopicMatch | None: the closest existing topic if within threshold, else None.
    """
    collection = registry.get()
    if collection.count() == 0:
        return None

    response = collection.query(
        query_texts=[topic], n_results=1, include=["documents", "distances"]
    )
    documents = response["documents"][0] if response["documents"] else []
    distances = response["distances"][0] if response["distances"] else []
    ids = response["ids"][0] if response["ids"] else []
    if not documents or distances[0] > threshold:
        return None

    return {"collection_name": ids[0], "topic": documents[0], "distance": distances[0]}


def find_collection_name(collection_name: str, registry: VectorStore) -> str | None:
    """
    Looks up the topic already registered under an exact `collection_name`.

    Args:
        collection_name (str): collection name to check.
        registry (VectorStore): the topics registry collection.

    Returns:
        str | None: the topic text registered under that id, or None if unused.
    """
    result = registry.get().get(ids=[collection_name], include=["documents"])
    documents = result["documents"] or []
    return documents[0] if documents else None


def register_topic(
    topic: str, collection_name: str, metadata: TopicMetadata, registry: VectorStore
) -> None:
    """
    Records a newly-ingested topic in the registry.

    Args:
        topic (str): topic text that was ingested.
        collection_name (str): collection name it was ingested into.
        registry (VectorStore): the topics registry collection.
    """

    registry.add(ids=[collection_name], documents=[topic], metadatas=[metadata])


def list_topics(registry: VectorStore) -> list[dict]:
    """
    Enumerates every topic currently registered.

    Args:
        registry (VectorStore): the topics registry collection.

    Returns:
        list[dict]: one dict per topic, shaped
            {"collection_name": str, "topic": str, **TopicMetadata}.
    """
    result = registry.get().get(include=["documents", "metadatas"])
    ids = result["ids"] or []
    documents = result["documents"] or []
    metadatas = result["metadatas"] or []
    return [
        {"collection_name": id_, "topic": document, **cast(TopicMetadata, metadata)}
        for id_, document, metadata in zip(ids, documents, metadatas)
    ]
