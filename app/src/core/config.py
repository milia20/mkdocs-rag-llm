"""
Конфигурация приложения.

Модуль содержит настройки приложения, загружаемые из переменных окружения.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QdrantConfig:
    """Configuration for Qdrant connection."""

    url: str
    collection_name: str
    api_key: str | None = None
    in_memory: bool = False
    path: str | None = None


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM provider."""

    provider: str
    model: str
    base_url: str | None
    temperature: float = 0.1
    max_tokens: int = 1024


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding service."""

    model_name: str
    dimension: int


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for search operations."""

    top_k_default: int = 5
    top_k_max: int = 50
    hybrid_weights: dict[str, float] | None = None

    def __post_init__(self) -> None:
        """Validate hybrid weights."""
        if self.hybrid_weights is None:
            object.__setattr__(self, "hybrid_weights", {"bm25": 0.5, "dense": 0.5})


class Settings(BaseSettings):
    """
    Настройки приложения.

    Атрибуты:
        app_name: Имя приложения.
        app_version: Версия приложения.
        debug: Режим отладки.
        qdrant_url: URL подключения к Qdrant.
        qdrant_collection: Имя коллекции Qdrant.
        embedding_model: Модель для эмбеддингов.
        llm_provider: Провайдер LLM.
        llm_model: Модель LLM.
        api_host: Хост API сервера.
        api_port: Порт API сервера.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Основные настройки
    app_name: str = Field(default="MkDocs RAG API", description="Имя приложения")
    app_version: str = Field(default="0.1.0", description="Версия приложения")
    debug: bool = Field(default=False, description="Режим отладки")

    # Настройки Qdrant
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="URL подключения к Qdrant",
    )
    qdrant_collection: str = Field(
        default="docs_index",
        description="Имя коллекции Qdrant",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="API ключ для Qdrant (опционально)",
    )
    qdrant_in_memory: bool = Field(
        default=False,
        description="Использовать in-memory режим Qdrant (для разработки)",
    )
    qdrant_path: str | None = Field(
        default=None,
        description="Путь к локальному хранилищу Qdrant",
    )

    # Настройки эмбеддингов
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Модель для генерации эмбеддингов",
    )
    embedding_dimension: int = Field(
        default=384,
        description="Размерность векторов эмбеддингов",
    )

    # Настройки LLM
    llm_provider: str = Field(
        default="ollama",
        description="Провайдер LLM (ollama, lmstudio)",
    )
    llm_model: str = Field(
        default="qwen2.5:7b",
        description="Модель LLM",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Базовый URL для LLM провайдера",
    )

    # Настройки API
    api_host: str = Field(default="0.0.0.0", description="Хост API сервера")
    api_port: int = Field(default=8000, ge=1, le=65535, description="Порт API сервера")
    cors_origins: list[str] = Field(
        default=["*"],
        description="Список разрешенных CORS origin",
    )

    # Настройки чанков
    chunk_max_tokens: int = Field(
        default=512,
        ge=100,
        le=2000,
        description="Максимальное количество токенов в чанке",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Перекрытие между чанками",
    )

    # Настройки гибридного поиска
    hybrid_weights: dict[str, float] = Field(
        default={"bm25": 0.5, "dense": 0.5},
        description="Веса для гибридного поиска",
    )

    # Таймауты и повторные попытки
    request_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Таймаут запроса в секундах",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Максимальное количество повторных попыток",
    )

    def get_qdrant_config(self) -> QdrantConfig:
        """
        Получить конфигурацию для подключения к Qdrant.

        Returns:
            QdrantConfig с параметрами подключения.
        """
        return QdrantConfig(
            url=self.qdrant_url,
            collection_name=self.qdrant_collection,
            api_key=self.qdrant_api_key,
            in_memory=self.qdrant_in_memory,
            path=self.qdrant_path,
        )

    def get_llm_config(self) -> LLMConfig:
        """
        Получить конфигурацию для LLM клиента.

        Returns:
            LLMConfig с параметрами LLM.
        """
        return LLMConfig(
            provider=self.llm_provider,
            model=self.llm_model,
            base_url=self.llm_base_url,
        )

    def get_embedding_config(self) -> EmbeddingConfig:
        """
        Получить конфигурацию для embedding сервиса.

        Returns:
            EmbeddingConfig с параметрами эмбеддингов.
        """
        return EmbeddingConfig(
            model_name=self.embedding_model,
            dimension=self.embedding_dimension,
        )

    def get_search_config(self) -> SearchConfig:
        """
        Получить конфигурацию для поиска.

        Returns:
            SearchConfig с параметрами поиска.
        """
        return SearchConfig(
            hybrid_weights=self.hybrid_weights,
        )


@lru_cache
def get_settings() -> Settings:
    """
    Получить кэшированный экземпляр настроек.

    Returns:
        Экземпляр Settings.
    """
    return Settings()


# Глобальный экземпляр настроек
settings = get_settings()
