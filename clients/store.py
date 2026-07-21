import os
from typing import Mapping, Sequence

from chromadb import CloudClient, Collection


class VectorStore:
    """Wraps a Chroma Cloud collection used to store document chunks"""

    def __init__(self, collection_name: str):
        self._client = CloudClient(
            api_key=os.environ["CHROMA_API_KEY"],
            tenant=os.environ["CHROMA_TENANT_ID"],
            database=os.environ["CHROMA_DB_NAME"],
        )
        self._collection_name = collection_name
        self.collection: Collection = self._client.get_or_create_collection(
            collection_name, embedding_function=None
        )

    def clear(self) -> None:
        """Deletes every existing record and recreates an empty collection"""
        self._client.delete_collection(self._collection_name)
        self.collection = self._client.get_or_create_collection(
            self._collection_name, embedding_function=None
        )

    def get(self) -> Collection:
        return self.collection

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: Sequence[Mapping[str, object]],
    ) -> None:
        self.collection.add(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )
