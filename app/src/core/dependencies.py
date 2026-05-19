"""
Dependency providers for FastAPI application.

This module provides dependency injection functions for services,
enabling better testability and separation of concerns.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from src.core.config import Settings, get_settings
from src.services.embedding_service import EmbeddingService
from src.services.llm_client import LLMClient
from src.services.qdrant_service import QdrantService


def get_qdrant_service(settings: Annotated[Settings, Depends(get_settings)]) -> QdrantService:
    """
    Dependency provider for Qdrant service.

    Args:
        settings: Application settings injected via Depends.

    Returns:
        Configured QdrantService instance.
    """
    return QdrantService(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        api_key=settings.qdrant_api_key,
    )


def get_embedding_service(settings: Annotated[Settings, Depends(get_settings)]) -> EmbeddingService:
    """
    Dependency provider for embedding service.

    Args:
        settings: Application settings injected via Depends.

    Returns:
        Configured EmbeddingService instance.
    """
    return EmbeddingService(model_name=settings.embedding_model)


def get_llm_client(settings: Annotated[Settings, Depends(get_settings)]) -> LLMClient:
    """
    Dependency provider for LLM client.

    Args:
        settings: Application settings injected via Depends.

    Returns:
        Configured LLMClient instance.
    """
    return LLMClient(
        base_url=settings.llm_base_url,
        model_name=settings.llm_model,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )


# Type aliases for cleaner endpoint signatures
QdrantServiceDep = Annotated[QdrantService, Depends(get_qdrant_service)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]
