"""
FastAPI application for RAG system.

Main application module that creates FastAPI instance
and registers all routes and middleware.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import chat, health, index, models, query, search
from src.core.config import settings
from src.middleware.error_handler import register_exception_handlers
from src.services.embedding_service import get_embedding_service
from src.services.llm_client import get_llm_client
from src.services.qdrant_service import get_qdrant_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan context manager.

    Performs initialization on startup and cleanup on shutdown.

    Args:
        app: FastAPI application instance.

    Yields:
        None
    """
    # Startup initialization
    logger.info("RAG API: starting application")
    logger.info("RAG API: initializing services...")

    qdrant_service = None
    llm_client = None

    try:
        # Initialize services
        qdrant_service = get_qdrant_service(
            in_memory=settings.qdrant_in_memory,
            path=settings.qdrant_path,
        )
        qdrant_service.connect()
        logger.info("Qdrant service initialized")

        embedder = get_embedding_service()
        logger.info(f"Embedding service initialized: {settings.embedding_model}")

        llm_client = get_llm_client()
        logger.info(f"LLM client initialized: {settings.llm_model}")

        yield

    finally:
        # Shutdown cleanup
        logger.info("RAG API: shutting down application")

        # Close connections
        if qdrant_service:
            try:
                qdrant_service.close()
            except Exception as e:
                logger.error(f"Error closing Qdrant connection: {e}")

        if llm_client:
            try:
                await llm_client.close()
            except Exception as e:
                logger.error(f"Error closing LLM client: {e}")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application instance.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="MkDocs RAG API",
        description="API for semantic search in MkDocs documentation using RAG",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register exception handlers
    register_exception_handlers(app)

    # Register routers
    app.include_router(query.router, prefix="/api/v1", tags=["query"])
    app.include_router(search.router, prefix="/api/v1", tags=["search"])
    app.include_router(index.router, prefix="/api/v1", tags=["index"])
    app.include_router(models.router, prefix="/api/v1", tags=["models"])
    app.include_router(chat.router, prefix="", tags=["chat"])
    app.include_router(health.router, prefix="", tags=["health"])  # Health endpoints at root level

    @app.get("/")
    async def root() -> dict[str, str]:
        """
        Root endpoint.

        Returns:
            Welcome message.
        """
        return {
            "message": "Welcome to MkDocs RAG API",
            "docs": "/docs",
            "health": "/health",
            "query": "/api/v1/query",
            "index": "/api/v1/index",
            "models": "/api/v1/models",
        }

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
