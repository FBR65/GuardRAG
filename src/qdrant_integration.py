"""
Qdrant Integration Module
Handles vector storage and retrieval using Qdrant vector database.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

logger = logging.getLogger(__name__)


@dataclass
class QdrantConfig:
    """Configuration for Qdrant client."""

    host: str = "localhost"
    port: int = 6333
    url: Optional[str] = None
    api_key: Optional[str] = None
    collection_name: str = "guardrag_documents"
    vector_size: int = 128  # COLPALI embedding dimension
    distance: Distance = Distance.COSINE


class QdrantVectorStore:
    """
    Qdrant vector store for COLPALI embeddings.
    """

    def __init__(self, config: QdrantConfig):
        """Initialize Qdrant vector store."""
        self.config = config
        self.client = self._init_client()
        self._ensure_collection()

    def _init_client(self) -> QdrantClient:
        """Initialize Qdrant client."""
        try:
            if self.config.url:
                client = QdrantClient(
                    url=self.config.url,
                    api_key=self.config.api_key,
                )
            else:
                client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                )

            # Test connection
            client.get_collections()
            logger.info("Qdrant client initialized successfully")
            return client

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def _ensure_collection(self):
        """Ensure collection exists."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.config.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.config.collection_name}")
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=self.config.distance,
                    ),
                )
            else:
                logger.info(f"Collection {self.config.collection_name} already exists")

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def add_embeddings(
        self,
        embeddings: List[np.ndarray],
        pages: List[Any],  # DocumentPage objects
        document_id: str,
    ) -> List[str]:
        """
        Add embeddings to Qdrant.

        Args:
            embeddings: List of COLPALI embeddings
            pages: List of document pages
            document_id: Unique document identifier

        Returns:
            List of point IDs
        """
        points = []
        point_ids = []

        for i, (embedding, page) in enumerate(zip(embeddings, pages)):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            # Convert embedding to list for Qdrant
            if isinstance(embedding, np.ndarray):
                vector = embedding.flatten().tolist()
            else:
                vector = embedding.flatten().tolist()

            # Prepare metadata
            metadata = {
                "document_id": document_id,
                "page_number": page.page_number,
                "text_content": page.text_content or "",
                "metadata": page.metadata or {},
            }

            # Create point
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=metadata,
            )
            points.append(point)

        try:
            # Upsert points
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points,
            )
            logger.info(f"Added {len(points)} embeddings to Qdrant")
            return point_ids

        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            document_id: Optional document ID filter

        Returns:
            List of search results
        """
        try:
            # Convert embedding to list
            if isinstance(query_embedding, np.ndarray):
                query_vector = query_embedding.flatten().tolist()
            else:
                query_vector = query_embedding.flatten().tolist()

            # Prepare filter
            query_filter = None
            if document_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                )

            # Search
            search_result = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

            # Format results
            results = []
            for hit in search_result:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                }
                results.append(result)

            logger.info(f"Found {len(results)} similar embeddings")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def delete_document(self, document_id: str) -> bool:
        """
        Delete all embeddings for a document.

        Args:
            document_id: Document ID to delete

        Returns:
            True if successful
        """
        try:
            # Delete points with matching document_id
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            logger.info(f"Deleted document {document_id} from Qdrant")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def clear_collection(self) -> bool:
        """Clear all embeddings from collection."""
        try:
            self.client.delete_collection(self.config.collection_name)
            self._ensure_collection()
            logger.info(f"Cleared collection {self.config.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        try:
            info = self.client.get_collection(self.config.collection_name)
            return {
                "name": info.name,
                "status": info.status,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "config": asdict(info.config),
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            collections = self.client.get_collections()
            collection_info = self.get_collection_info()

            return {
                "status": "healthy",
                "collections_count": len(collections.collections),
                "target_collection": collection_info,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
