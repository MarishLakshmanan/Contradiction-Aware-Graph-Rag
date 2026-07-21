"""Test for the topics registry"""

from datetime import datetime, timezone

from chromadb import CloudClient

from src.clients.store import VectorStore
from src.make import TOPIC_SIMILARITY_THRESHOLD
from src.schema import TopicMetadata
from src.utils.topics import find_collection_name, find_similar_topic, register_topic
from settings import settings

TOPIC_A = "Do larger language models actually reason better, or just pattern-match better?"
TOPIC_A_REWORDED = "Do bigger LLMs really reason more, or are they just better at pattern matching?"
TOPIC_UNRELATED = "How does photosynthesis work in C4 plants?"

TEST_METADATA: TopicMetadata = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "num_of_pdfs": 10,
    "max_chars": 1000,
    "chunk_overlap": 200,
}


def test_find_similar_topic(registry: VectorStore) -> None:
    register_topic(TOPIC_A, "topic-a", TEST_METADATA, registry)
    register_topic(TOPIC_UNRELATED, "topic-unrelated", TEST_METADATA, registry)

    match = find_similar_topic(TOPIC_A_REWORDED, registry, TOPIC_SIMILARITY_THRESHOLD)
    assert match is not None, "expected a match for a reworded near-duplicate topic"
    assert match["collection_name"] == "topic-a", (
        f"expected match against topic-a, got {match['collection_name']!r} "
        f"(distance={match['distance']})"
    )
    print(
        f"[similar] OK - reworded topic matched topic-a (distance={match['distance']:.4f})"
    )

    no_match = find_similar_topic(
        "What's the best pizza topping combination?", registry, TOPIC_SIMILARITY_THRESHOLD
    )
    assert no_match is None, f"expected no match for an unrelated topic, got {no_match}"
    print("[unrelated] OK - unrelated topic did not match")


def test_find_collection_name(registry: VectorStore) -> None:
    topic = find_collection_name("topic-a", registry)
    assert topic == TOPIC_A, f"expected {TOPIC_A!r}, got {topic!r}"

    missing = find_collection_name("nonexistent-collection", registry)
    assert missing is None, f"expected None for an unused collection_name, got {missing!r}"

    print("[collection_name] OK - exact id lookup found and missed correctly")


if __name__ == "__main__":
    collection_name = "topics_test"
    client = CloudClient(
        api_key=settings.chroma_api_key,
        tenant=settings.chroma_tenant_id,
        database=settings.chroma_db_name,
    )

    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)

    registry = VectorStore(collection_name=collection_name)

    # Run all the tests
    test_find_similar_topic(registry)
    test_find_collection_name(registry)

    client.delete_collection(collection_name)
