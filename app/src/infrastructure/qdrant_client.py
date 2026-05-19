"""
Low-level Qdrant client wrapper.

This module provides a thin wrapper around the Qdrant Python SDK,
supporting both remote connections and local in-memory/disk modes.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.core.exceptions import QdrantError

logger = logging.getLogger(__name__)


class QdrantClientWrapper:
    """
    Low-level wrapper for Qdrant client.

    This class handles connection management and raw operations.
    For business logic, use the VectorStore service.

    Attributes:
        url: URL for remote Qdrant instance.
        collection_name: Default collection name.
        api_key: API key for authentication.
        in_memory: Use in-memory mode (for testing).
        path: Path for local disk storage.
    """

    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
        api_key: str | None = None,
        in_memory: bool = False,
        path: str | None = None,
    ) -> None:
        """
        Initialize Qdrant client wrapper.

        Args:
            url: URL for Qdrant connection.
            collection_name: Default collection name.
            api_key: API key for authentication.
            in_memory: Use in-memory mode.
            path: Path for local disk storage.
        """
        self.url = url
        self.collection_name = collection_name or "default_collection"
        self.api_key = api_key
        self.in_memory = in_memory
        self.path = path

        logger.info(
            f"Initializing QdrantClientWrapper: url={url or 'N/A'}, "
            f"collection={self.collection_name}, in_memory={in_memory}, path={path}"
        )

        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        """
        Get or create Qdrant client instance (lazy initialization).

        Returns:
            QdrantClient instance.

        Raises:
            QdrantError: If connection fails.
        """
        if self._client is None:
            try:
                if self.in_memory:
                    self._client = QdrantClient(":memory:")
                    logger.info("Qdrant initialized in in-memory mode")
                elif self.path:
                    self._client = QdrantClient(path=self.path)
                    logger.info(f"Qdrant initialized in local disk mode: {self.path}")
                else:
                    if not self.url:
                        msg = "URL is required for remote Qdrant connection"
                        raise ValueError(msg)
                    self._client = QdrantClient(
                        url=self.url,
                        api_key=self.api_key,
                    )
                    # Verify connection
                    self._client.get_collections()
                    logger.info("Successfully connected to remote Qdrant")
            except Exception as e:
                msg = f"Failed to connect to Qdrant: {e}"
                logger.error(msg)
                raise QdrantError(msg, str(e)) from e
        return self._client

    def create_collection(
        self,
        collection_name: str | None = None,
        vector_size: int = 384,
        distance: str = "Cosine",
    ) -> bool:
        """
        Create a collection in Qdrant.

        Args:
            collection_name: Collection name (uses default if not provided).
            vector_size: Dimension of vectors.
            distance: Distance function (Cosine, Euclid, Dot).

        Returns:
            True if collection created or already exists.

        Raises:
            QdrantError: If creation fails.
        """
        name = collection_name or self.collection_name

        try:
            collections = self.client.get_collections().collections
            existing = any(c.name == name for c in collections)

            if existing:
                logger.info(f"Collection '{name}' already exists")
                return True

            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance[distance.upper()],
                ),
            )
            logger.info(f"Collection '{name}' created successfully")
            return True

        except Exception as e:
            msg = f"Error creating collection: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    def upsert_points(
        self,
        points: list[PointStruct],
        collection_name: str | None = None,
    ) -> bool:
        """
        Add or update points in a collection.

        Args:
            points: List of points to add.
            collection_name: Collection name (uses default if not provided).

        Returns:
            True if operation successful.

        Raises:
            QdrantError: If upsert fails.
        """
        name = collection_name or self.collection_name

        try:
            result = self.client.upsert(
                collection_name=name,
                points=points,
            )
            logger.info(f"Upserted {len(points)} points to collection '{name}'")
            return result.status == "completed"

        except Exception as e:
            msg = f"Error upserting points: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector.
            limit: Maximum number of results.
            filter_dict: Filter for search (optional).
            collection_name: Collection name (uses default if not provided).

        Returns:
            List of search results.

        Raises:
            QdrantError: If search fails.
        """
        name = collection_name or self.collection_name

        try:
            results = self.client.search(
                collection_name=name,
                query_vector=query_vector,
                query_filter=filter_dict,
                limit=limit,
            )

            return [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload,
                }
                for point in results
            ]

        except Exception as e:
            msg = f"Search error: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    def delete_collection(self, collection_name: str | None = None) -> bool:
        """
        Delete a collection.

        Args:
            collection_name: Collection name (uses default if not provided).

        Returns:
            True if collection deleted.

        Raises:
            QdrantError: If deletion fails.
        """
        name = collection_name or self.collection_name

        try:
            self.client.delete_collection(collection_name=name)
            logger.info(f"Collection '{name}' deleted")
            return True

        except Exception as e:
            msg = f"Error deleting collection: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    def close(self) -> None:
        """Close the Qdrant client connection."""
        if self._client:
            self._client.close()
            logger.info("Qdrant client connection closed")
            self._client = None
