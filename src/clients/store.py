from typing import Mapping, Sequence, cast

from chromadb import CloudClient, Collection
from chromadb.api.types import Embeddable, EmbeddingFunction

from settings import settings
from src.clients.embed import HuggingFaceRouterEmbeddingFunction


class VectorStore:
    """Wraps a Chroma Cloud collection used to store document chunks."""

    def __init__(self, collection_name: str):
        """
        Connects to Chroma Cloud and gets or creates the named collection.

        Args:
            collection_name (str): name of the Chroma collection to use.
        """
        self._client = CloudClient(
            api_key=settings.chroma_api_key,
            tenant=settings.chroma_tenant_id,
            database=settings.chroma_db_name,
        )
        self._collection_name = collection_name
        # chromadb's own EmbeddingFunction[Documents] subclasses (including its
        # built-ins) don't structurally satisfy get_or_create_collection's
        # EmbeddingFunction[Embeddable] parameter -- Documents isn't a supertype
        # of Embeddable, which the Protocol's contravariant D requires. Cast once
        # here rather than at every call site.
        self._embedding_function = cast(
            EmbeddingFunction[Embeddable],
            HuggingFaceRouterEmbeddingFunction(model_name=settings.embedding_model_id),
        )
        self.collection: Collection = self._client.get_or_create_collection(
            collection_name, embedding_function=self._embedding_function
        )

    def clear(self) -> None:
        """Deletes every existing record and recreates an empty collection."""
        self._client.delete_collection(self._collection_name)
        self.collection = self._client.get_or_create_collection(
            self._collection_name, embedding_function=self._embedding_function
        )

    def get(self) -> Collection:
        """
        Returns the underlying Chroma collection.

        Returns:
            Collection: the wrapped Chroma collection.
        """
        return self.collection

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: Sequence[Mapping[str, object]],
    ) -> None:
        """
        Adds records to the collection.

        Args:
            ids (list[str]): unique id for each record.
            documents (list[str]): text content for each record, embedded automatically.
            metadatas (Sequence[Mapping[str, object]]): metadata dict for each record.
        """
        # pyright can't verify a TypedDict (ChunkMetadata, PaperMetadata, ...) against
        # chromadb's Metadata Mapping[str, <narrow scalar union>] -- it widens a
        # TypedDict's value type to `object` for this check regardless of its actual
        # field types, so this is a pyright limitation, not a real type hole.
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)  # type: ignore[arg-type]
