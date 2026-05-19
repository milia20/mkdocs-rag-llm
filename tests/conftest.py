"""
Фикстуры для тестирования.

Модуль содержит общие фикстуры pytest для тестов.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_config() -> dict[str, Any]:
    """
    Конфигурация для тестов.

    Returns:
        Словарь с тестовой конфигурацией.
    """
    return {
        "qdrant_url": ":memory:",  # Use in-memory mode for tests
        "qdrant_collection": "test_docs_index",
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "llm_provider": "ollama",
        "llm_model": "qwen2.5:7b",
        "api_host": "localhost",
        "api_port": 8001,  # Отличный от основного порт для тестов
        "enable_chat_panel": True,
        "api_password": "",
        "qdrant_in_memory": True,  # Flag to enable in-memory mode
    }


@pytest.fixture(scope="session")
def sample_html() -> str:
    """
    Пример HTML-контента для тестов.

    Returns:
        Строка с HTML-разметкой.
    """
    return """
    <html>
        <head><title>Тестовая страница</title></head>
        <body>
            <h1>Заголовок</h1>
            <p>Это тестовый абзац с <b>жирным</b> текстом.</p>
            <script>alert('should be removed');</script>
            <style>.hidden { display: none; }</style>
            <a href="https://example.com">Ссылка</a>
        </body>
    </html>
    """


@pytest.fixture(scope="session")
def sample_markdown() -> str:
    """
    Пример Markdown-контента для тестов.

    Returns:
        Строка с Markdown-разметкой.
    """
    return """
# Заголовок первого уровня

## Заголовок второго уровня

Это обычный текст с **жирным** и *курсивным* форматированием.

[Ссылка](https://example.com)

```python
def hello():
    print("Привет, мир!")
```

- Список 1
- Список 2
- Список 3
"""


@pytest.fixture
def temp_dir(tmp_path: Path) -> Generator[Path]:
    """
    Временная директория для тестов.

    Args:
        tmp_path: Фикстура pytest для временных путей.

    Yields:
        Путь к временной директории.
    """
    test_dir = tmp_path / "rag_test"
    test_dir.mkdir()
    yield test_dir


@pytest.fixture
def mock_qdrant_response() -> list[dict[str, Any]]:
    """
    Мок ответа от Qdrant.

    Returns:
        Список мок-результатов поиска.
    """
    return [
        {
            "id": "doc_1_chunk_0",
            "score": 0.95,
            "payload": {
                "text": "Тестовый документ 1",
                "source": "test_doc.md",
                "page": "index",
            },
        },
        {
            "id": "doc_2_chunk_0",
            "score": 0.87,
            "payload": {
                "text": "Тестовый документ 2",
                "source": "another_doc.md",
                "page": "about",
            },
        },
    ]


@pytest.fixture
def mock_embedding_vector() -> list[float]:
    """
    Мок вектора эмбеддинга.

    Returns:
        Список float чисел, имитирующий вектор.
    """
    return [0.1] * 384  # Стандартная размерность для MiniLM


@pytest.fixture
def app_client() -> Generator[TestClient]:
    """
    Тестовый клиент FastAPI приложения.

    Yields:
        TestClient для тестирования API.
    """
    from src.main import app
    # Initialize Qdrant service in memory mode before creating the client
    from src.services.qdrant_service import get_qdrant_service

    get_qdrant_service(in_memory=True)

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def mock_services(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """
    Mock all external services for API testing.

    This fixture mocks Qdrant, embedding service, and LLM client
    to prevent actual network calls during tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Dictionary containing mock objects.
    """
    from unittest.mock import AsyncMock, MagicMock, Mock

    # Mock Qdrant service
    mock_qdrant = MagicMock()
    mock_qdrant.connect = Mock(return_value=None)
    mock_qdrant.search = Mock(return_value=[])
    mock_qdrant.close = Mock(return_value=None)

    # Mock embedding service
    mock_embedder = MagicMock()
    mock_embedder.encode_query = Mock(return_value=[0.1] * 384)
    mock_embedder.encode_batch = Mock(return_value=[[0.1] * 384])

    # Mock LLM client
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="Test response")
    mock_llm.generate_stream = AsyncMock(return_value=iter(["Test ", "response"]))
    mock_llm.close = AsyncMock(return_value=None)
    mock_llm.model_name = "test-model"

    # Patch the singleton getters
    monkeypatch.setattr(
        "src.services.qdrant_service.get_qdrant_service",
        Mock(return_value=mock_qdrant),
    )
    monkeypatch.setattr(
        "src.services.embedding_service.get_embedding_service",
        Mock(return_value=mock_embedder),
    )
    monkeypatch.setattr(
        "src.services.llm_client.get_llm_client",
        Mock(return_value=mock_llm),
    )

    return {
        "qdrant": mock_qdrant,
        "embedder": mock_embedder,
        "llm": mock_llm,
    }


@pytest.fixture(autouse=True)
def set_log_level() -> None:
    """
    Установить уровень логирования для тестов.

    Автоматически применяется ко всем тестам.
    """
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("mkdocs.plugins.rag_plugin").setLevel(logging.DEBUG)
    logging.getLogger("src").setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def test_env_vars(monkeypatch: pytest.MonkeyPatch, test_config: dict[str, Any]) -> None:
    """
    Set environment variables for testing.

    This fixture sets all RAG_* environment variables to test values
    before any tests run, preventing connection attempts to real services.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        test_config: Test configuration dictionary.
    """
    # Note: qdrant_url is set to ":memory:" but won't be used since we initialize
    # the service with in_memory=True directly in the fixtures
    monkeypatch.setenv(
        "RAG_QDRANT_URL", "http://localhost:6333"
    )  # Dummy value, not used in memory mode
    monkeypatch.setenv("RAG_QDRANT_COLLECTION", test_config["qdrant_collection"])
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", test_config["embedding_model"])
    monkeypatch.setenv("RAG_LLM_PROVIDER", test_config["llm_provider"])
    monkeypatch.setenv("RAG_LLM_MODEL", test_config["llm_model"])
    monkeypatch.setenv("RAG_API_HOST", test_config["api_host"])
    monkeypatch.setenv("RAG_API_PORT", str(test_config["api_port"]))
