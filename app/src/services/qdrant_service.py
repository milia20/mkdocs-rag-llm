"""
Сервис для работы с векторной базой данных Qdrant.

Модуль предоставляет класс для подключения, индексации и поиска
в векторной базе данных Qdrant.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client.models import PointStruct
from src.core.config import settings
from src.infrastructure.qdrant_client import QdrantClientWrapper

logger = logging.getLogger(__name__)


class QdrantService:
    """
    Сервис для взаимодействия с Qdrant.

    Предоставляет методы для создания коллекций, индексации документов
    и выполнения семантического поиска.

    This service wraps the low-level QdrantClientWrapper and provides
    business logic for vector operations.

    Attributes:
        client: Low-level Qdrant client wrapper.
        collection_name: Имя коллекции для работы.
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
        Инициализация сервиса Qdrant.

        Args:
            url: URL подключения к Qdrant. Если не указан, используется из настроек.
            collection_name: Имя коллекции. Если не указано, используется из настроек.
            api_key: API ключ для подключения. Если не указан, используется из настроек.
            in_memory: Использовать режим в памяти (для тестирования).
            path: Путь к директории для хранения данных (для локального режима).
        """
        # Use provided values or fall back to settings
        resolved_url = url or settings.qdrant_url
        resolved_collection = collection_name or settings.qdrant_collection
        resolved_api_key = api_key or settings.qdrant_api_key

        logger.info(
            f"Инициализация QdrantService: url={resolved_url}, collection={resolved_collection}, "
            f"in_memory={in_memory}, path={path}"
        )

        # Use low-level client wrapper
        self.client = QdrantClientWrapper(
            url=resolved_url if not in_memory else None,
            collection_name=resolved_collection,
            api_key=resolved_api_key if not in_memory else None,
            in_memory=in_memory,
            path=path,
        )
        self.collection_name = resolved_collection

    def connect(self) -> None:
        """
        Установить подключение к Qdrant.

        This method triggers lazy initialization of the underlying client.

        Raises:
            QdrantError: Если не удалось подключиться.
        """
        # Access the client property to trigger initialization
        _ = self.client.client
        logger.info("Qdrant connection established")

    def create_collection(
        self,
        vector_size: int = 384,
        distance: str = "Cosine",
    ) -> bool:
        """
        Создать коллекцию в Qdrant.

        Args:
            vector_size: Размерность векторов.
            distance: Функция расстояния (Cosine, Euclid, Dot).

        Returns:
            True если коллекция создана или уже существует.

        Raises:
            QdrantError: Если произошла ошибка при создании.
        """
        return self.client.create_collection(
            vector_size=vector_size,
            distance=distance,
        )

    def upsert_points(
        self,
        points: list[PointStruct],
    ) -> bool:
        """
        Добавить или обновить точки в коллекции.

        Args:
            points: Список точек для добавления.

        Returns:
            True если операция успешна.

        Raises:
            QdrantError: Если произошла ошибка при добавлении.
        """
        return self.client.upsert_points(points=points)

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Выполнить поиск похожих векторов.

        Args:
            query_vector: Вектор запроса.
            limit: Максимальное количество результатов.
            filter_dict: Фильтр для поиска (опционально).

        Returns:
            Список результатов поиска.

        Raises:
            QdrantError: Если произошла ошибка при поиске.
        """
        return self.client.search(
            query_vector=query_vector,
            limit=limit,
            filter_dict=filter_dict,
        )

    def delete_collection(self) -> bool:
        """
        Удалить коллекцию.

        Returns:
            True если коллекция удалена.

        Raises:
            QdrantError: Если произошла ошибка при удалении.
        """
        return self.client.delete_collection()

    def close(self) -> None:
        """Закрыть подключение к Qdrant."""
        self.client.close()


# Singleton instance
_qdrant_service: QdrantService | None = None


def get_qdrant_service(in_memory: bool = False, path: str | None = None) -> QdrantService:
    """
    Получить экземпляр сервиса Qdrant (singleton).

    Args:
        in_memory: Использовать режим в памяти (для тестирования).
        path: Путь к директории для хранения данных (для локального режима).

    Returns:
        Экземпляр QdrantService.
    """
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService(in_memory=in_memory, path=path)
    return _qdrant_service
