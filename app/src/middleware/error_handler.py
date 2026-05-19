"""
Middleware for error handling.

This module provides centralized exception handling for the API.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse
from src.core.exceptions import (ConfigurationError, DocumentProcessingError, EmbeddingError, LLMError, QdrantError,
                                 RAGException, SearchError)

logger = logging.getLogger(__name__)


async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
    """
    Handle RAG exceptions and return formatted error response.

    Args:
        request: The incoming request.
        exc: The RAGException that was raised.

    Returns:
        JSONResponse with error details.
    """
    logger.error(f"RAG exception: {exc.__class__.__name__} - {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
            "path": str(request.url.path),
        },
    )


async def configuration_error_handler(request: Request, exc: ConfigurationError) -> JSONResponse:
    """Handle configuration errors."""
    logger.error(f"Configuration error: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "ConfigurationError",
            "message": exc.message,
            "details": exc.details,
        },
    )


async def qdrant_error_handler(request: Request, exc: QdrantError) -> JSONResponse:
    """Handle Qdrant errors."""
    logger.error(f"Qdrant error: {exc}")

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "QdrantError",
            "message": exc.message,
            "details": exc.details,
        },
    )


async def embedding_error_handler(request: Request, exc: EmbeddingError) -> JSONResponse:
    """Handle embedding errors."""
    logger.error(f"Embedding error: {exc}")

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "EmbeddingError",
            "message": exc.message,
            "details": exc.details,
        },
    )


async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    """Handle LLM errors."""
    logger.error(f"LLM error: {exc}")

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "LLMError",
            "message": exc.message,
            "details": exc.details,
        },
    )


async def search_error_handler(request: Request, exc: SearchError) -> JSONResponse:
    """Handle search errors."""
    logger.error(f"Search error: {exc}")

    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "SearchError",
            "message": exc.message,
            "details": exc.details,
        },
    )


async def document_processing_error_handler(
    request: Request,
    exc: DocumentProcessingError,
) -> JSONResponse:
    """Handle document processing errors."""
    logger.error(f"Document processing error: {exc}")

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "DocumentProcessingError",
            "message": exc.message,
            "details": exc.details,
        },
    )


def register_exception_handlers(app: Any) -> None:
    """
    Register all exception handlers with the FastAPI app.

    Args:
        app: FastAPI application instance.
    """
    from fastapi import FastAPI

    if not isinstance(app, FastAPI):
        logger.warning("Cannot register exception handlers: not a FastAPI app")
        return

    app.add_exception_handler(RAGException, rag_exception_handler)
    app.add_exception_handler(ConfigurationError, configuration_error_handler)
    app.add_exception_handler(QdrantError, qdrant_error_handler)
    app.add_exception_handler(EmbeddingError, embedding_error_handler)
    app.add_exception_handler(LLMError, llm_error_handler)
    app.add_exception_handler(SearchError, search_error_handler)
    app.add_exception_handler(
        DocumentProcessingError,
        document_processing_error_handler,
    )
