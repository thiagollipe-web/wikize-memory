from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from wikize_memory.config import Settings
from wikize_memory.schemas import (
    ArticleAnalysis,
    ArticleDocument,
    ModelMetadata,
)
from wikize_memory.web import app as web_app
from wikize_memory.web.app import create_app

ADMIN_TOKEN = "test-admin-token-with-enough-entropy"
AUTH_HEADERS = {"Authorization": f"Bearer {ADMIN_TOKEN}"}


@pytest.fixture
def client(tmp_path) -> Iterator[TestClient]:
    app = create_app(
        database_path=tmp_path / "memories.sqlite3",
        admin_token=ADMIN_TOKEN,
        settings=Settings(user_agent="MemoriaVivaBotTest/1.0 (+https://example.org/contact)"),
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health_is_public_but_memory_routes_are_protected(client: TestClient) -> None:
    assert client.get("/api/healthz").status_code == 200
    assert client.get("/api/memories").status_code == 401
    assert client.get("/api/summary").status_code == 401
    assert client.post("/api/harvest", json={"title": "Memória"}).status_code == 401
    assert (
        client.get(
            "/api/memories?status=pending",
            headers={"Authorization": "Bearer invalid-token"},
        ).status_code
        == 401
    )


def test_import_review_and_export_round_trip(client: TestClient) -> None:
    pack = {
        "format": "thigas-synapse-memories",
        "version": 3,
        "title": "Importação de teste",
        "memories": [
            {
                "subject": "Hipocampo",
                "fact": "O hipocampo participa da consolidação de memórias.",
                "tags": ["neurociência", "memória"],
                "aliases": ["formação hipocampal"],
                "approved": False,
                "confidence": 0.8,
                "source": "https://pt.wikipedia.org/wiki/Hipocampo",
            }
        ],
    }
    imported = client.post("/api/import", headers=AUTH_HEADERS, json=pack)
    assert imported.status_code == 200
    assert imported.json() == {"imported": 1, "duplicates": 0}

    pending = client.get(
        "/api/memories?status=pending",
        headers=AUTH_HEADERS,
    )
    assert pending.status_code == 200
    assert len(pending.json()) == 1
    memory_id = pending.json()[0]["id"]

    approved_before = client.get("/api/memories", headers=AUTH_HEADERS)
    assert approved_before.status_code == 200
    assert approved_before.json() == []

    changed = client.patch(
        f"/api/memories/{memory_id}",
        headers=AUTH_HEADERS,
        json={"status": "approved"},
    )
    assert changed.status_code == 200
    assert changed.json()["status"] == "approved"

    approved_after = client.get("/api/memories", headers=AUTH_HEADERS)
    assert len(approved_after.json()) == 1
    exported = client.get("/api/export", headers=AUTH_HEADERS)
    assert exported.status_code == 200
    assert exported.json()["memories"][0]["approved"] is True


def test_import_deduplicates_case_and_whitespace(client: TestClient) -> None:
    first = {
        "format": "thigas-synapse-memories",
        "version": 3,
        "title": "Primeiro",
        "memories": [
            {
                "subject": "Hipocampo",
                "fact": "Armazena  memórias episódicas.",
                "tags": [],
                "aliases": [],
                "approved": False,
                "confidence": 0.5,
                "source": "teste",
            }
        ],
    }
    second = {
        **first,
        "title": "Segundo",
        "memories": [
            {
                **first["memories"][0],
                "subject": "  HIPOCAMPO  ",
                "fact": "armazena memórias episódicas.",
            }
        ],
    }
    assert client.post("/api/import", headers=AUTH_HEADERS, json=first).json() == {
        "imported": 1,
        "duplicates": 0,
    }
    assert client.post("/api/import", headers=AUTH_HEADERS, json=second).json() == {
        "imported": 0,
        "duplicates": 1,
    }


def test_private_read_routes_require_token(client: TestClient) -> None:
    assert client.get("/api/export").status_code == 401
    assert client.get("/api/export?include_pending=true").status_code == 401
    assert client.get("/api/summary", headers=AUTH_HEADERS).status_code == 200


def test_short_admin_token_keeps_service_unavailable(tmp_path) -> None:
    app = create_app(database_path=tmp_path / "memories.sqlite3", admin_token="short")
    with TestClient(app) as test_client:
        assert test_client.get("/api/healthz").status_code == 503
        assert test_client.get("/api/memories").status_code == 503


def test_harvest_saves_pending_memories(tmp_path, monkeypatch) -> None:
    article = ArticleDocument(
        requested_title="Hipocampo",
        title="Hipocampo",
        page_id=123,
        canonical_url="https://pt.wikipedia.org/wiki/Hipocampo",
        revision_id=456,
        revision_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        retrieved_at=datetime(2025, 1, 2, tzinfo=UTC),
        categories=["Neurociência"],
        content="O hipocampo participa da consolidação da memória.",
        original_text_characters=50,
        content_truncated=False,
    )
    analysis = ArticleAnalysis(
        extractive_summary="O hipocampo participa da consolidação da memória.",
        central_claims=["O hipocampo participa da consolidação da memória."],
        topics=["Neurociência", "Memória"],
        models=ModelMetadata(
            ner_model="test-model",
            abstractive_model=None,
            embedding_model=None,
            abstractive_summary_available=False,
            embeddings_available=False,
        ),
    )

    class FakeProcessor:
        def analyze(self, *_args, **_kwargs) -> ArticleAnalysis:
            return analysis

    async def fetch_article(*_args, **_kwargs) -> ArticleDocument:
        return article

    monkeypatch.setattr(web_app, "NLPProcessor", lambda _settings: FakeProcessor())
    monkeypatch.setattr(web_app, "_fetch_article", fetch_article)
    app = create_app(
        database_path=tmp_path / "memories.sqlite3",
        admin_token=ADMIN_TOKEN,
        settings=Settings(user_agent="MemoriaVivaBotTest/1.0 (+https://example.org/contact)"),
    )

    with TestClient(app) as test_client:
        queued = test_client.post(
            "/api/harvest",
            headers=AUTH_HEADERS,
            json={"title": "Hipocampo", "abstractive": False, "embeddings": False},
        )
        assert queued.status_code == 202
        job = test_client.get(f"/api/jobs/{queued.json()['id']}", headers=AUTH_HEADERS)
        assert job.status_code == 200
        assert job.json()["status"] == "completed"
        pending = test_client.get(
            "/api/memories?status=pending",
            headers=AUTH_HEADERS,
        )
        assert pending.status_code == 200
        assert any(
            memory["subject"] == "Hipocampo" and "consolidação da memória" in memory["fact"]
            for memory in pending.json()
        )
