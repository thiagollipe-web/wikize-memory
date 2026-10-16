"""API e interface web para colher, revisar e guardar memórias."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import sqlite3
import threading
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from wikize_memory.config import Settings
from wikize_memory.errors import (
    ConfigurationError,
    NLPDependencyError,
    NLPError,
    WikipediaDisambiguationError,
    WikipediaError,
    WikipediaNotFoundError,
)
from wikize_memory.ingestion import WikipediaClient
from wikize_memory.nlp import NLPProcessor
from wikize_memory.pipeline import build_harvest_result
from wikize_memory.schemas import HarvestResult, MemoryPack, MemoryRecord

PACKAGE_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = PACKAGE_ROOT / "static"
_analysis_lock = threading.Lock()
MIN_ADMIN_TOKEN_LENGTH = 32


class HarvestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=300)
    abstractive: bool = True
    embeddings: bool = True
    max_paragraphs: int = Field(default=8, ge=1, le=30)


class JobResponse(BaseModel):
    id: str
    title: str
    status: Literal["queued", "downloading", "analyzing", "saving", "completed", "failed"]
    message: str
    error: str | None = None
    result: dict[str, object] | None = None
    created_at: datetime
    updated_at: datetime


class MemoryStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pending", "approved", "rejected"]


class MemoryResponse(BaseModel):
    id: str
    subject: str
    fact: str
    tags: list[str]
    aliases: list[str]
    status: Literal["pending", "approved", "rejected"]
    confidence: float
    source: str
    source_url: str | None
    article_title: str
    created_at: datetime


class ImportResponse(BaseModel):
    imported: int
    duplicates: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _database_path() -> Path:
    configured = os.environ.get("WIKIZEMEMORY_DATABASE_PATH", "data/memories.sqlite3")
    return Path(configured).expanduser().resolve()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


@contextmanager
def _connection(path: Path):
    connection = _connect(path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(path: Path) -> None:
    with _connection(path) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                error TEXT,
                result_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL UNIQUE,
                subject TEXT NOT NULL,
                fact TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending', 'approved', 'rejected')),
                confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                source TEXT NOT NULL,
                source_url TEXT,
                article_title TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_status_created
                ON memories(status, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_subject
                ON memories(subject COLLATE NOCASE);
            CREATE INDEX IF NOT EXISTS idx_jobs_created
                ON jobs(created_at DESC);
            """
        )


def _fingerprint(record: MemoryRecord) -> str:
    subject = " ".join(record.subject.casefold().split())
    fact = " ".join(record.fact.casefold().split())
    return hashlib.sha256(f"{subject}\n{fact}".encode()).hexdigest()


def _insert_memory(
    connection: sqlite3.Connection,
    record: MemoryRecord,
    *,
    status: Literal["pending", "approved", "rejected"],
    article_title: str,
    source_url: str | None,
) -> bool:
    cursor = connection.execute(
        """
        INSERT OR IGNORE INTO memories (
            id, fingerprint, subject, fact, tags_json, aliases_json, status,
            confidence, source, source_url, article_title, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            _fingerprint(record),
            record.subject,
            record.fact,
            json.dumps(record.tags, ensure_ascii=False),
            json.dumps(record.aliases, ensure_ascii=False),
            status,
            record.confidence,
            record.source,
            source_url,
            article_title,
            _now(),
        ),
    )
    return cursor.rowcount == 1


def _memory_from_row(row: sqlite3.Row) -> MemoryResponse:
    return MemoryResponse(
        id=row["id"],
        subject=row["subject"],
        fact=row["fact"],
        tags=json.loads(row["tags_json"]),
        aliases=json.loads(row["aliases_json"]),
        status=row["status"],
        confidence=row["confidence"],
        source=row["source"],
        source_url=row["source_url"],
        article_title=row["article_title"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def create_app(
    *,
    database_path: Path | None = None,
    admin_token: str | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Cria uma instância configurável, também usada pelos testes."""
    db_path = (database_path or _database_path()).resolve()
    token = (
        admin_token if admin_token is not None else os.environ.get("WIKIZEMEMORY_ADMIN_TOKEN", "")
    )
    app_settings = settings or Settings()
    processor = NLPProcessor(app_settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        initialize_database(db_path)
        # Tarefas em memória não sobrevivem a um restart. Registra isso em vez de
        # deixar trabalhos antigos presos indefinidamente em "queued"/"analyzing".
        with _connection(db_path) as connection:
            connection.execute(
                """UPDATE jobs SET status = 'failed',
                   message = 'A execução foi interrompida por uma reinicialização.',
                   error = 'Interrompido durante a execução; envie o artigo novamente.',
                   updated_at = ? WHERE status IN (
                       'queued', 'downloading', 'analyzing', 'saving'
                   )""",
                (_now(),),
            )
        yield

    app = FastAPI(
        title="Memórias da Wikipédia PT",
        description=(
            "Bot para extrair, revisar e guardar memórias de artigos da Wikipédia em português."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.database_path = db_path
    app.state.admin_token = token
    app.state.settings = app_settings
    app.state.processor = processor
    app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

    def require_admin_token(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        expected = app.state.admin_token.strip()
        if len(expected) < MIN_ADMIN_TOKEN_LENGTH:
            raise HTTPException(
                status_code=503,
                detail=(
                    "WIKIZEMEMORY_ADMIN_TOKEN precisa conter pelo menos "
                    f"{MIN_ADMIN_TOKEN_LENGTH} caracteres."
                ),
            )
        parts = authorization.split(" ", maxsplit=1) if authorization else []
        supplied = parts[1].strip() if len(parts) == 2 and parts[0].casefold() == "bearer" else ""
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="Acesso não autorizado.")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC_ROOT / "index.html")

    @app.get("/api/healthz")
    def health() -> dict[str, str]:
        if len(app.state.admin_token.strip()) < MIN_ADMIN_TOKEN_LENGTH:
            raise HTTPException(
                status_code=503,
                detail="O serviço ainda não tem uma chave de acesso válida configurada.",
            )
        return {"status": "ok", "service": "memorias-wikipedia-pt"}

    @app.post("/api/harvest", response_model=JobResponse, status_code=202)
    def harvest(
        payload: HarvestRequest,
        background_tasks: BackgroundTasks,
        _authorized: None = Depends(require_admin_token),
    ) -> JobResponse:
        title = payload.title.strip()
        if not title:
            raise HTTPException(status_code=422, detail="Informe o título de um artigo.")
        job_id = str(uuid4())
        created_at = _now()
        message = "Na fila para consultar a Wikipédia."
        with _connection(db_path) as connection:
            connection.execute(
                "INSERT INTO jobs (id, title, status, message, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, title, "queued", message, created_at, created_at),
            )
        background_tasks.add_task(
            _run_harvest,
            db_path,
            job_id,
            title,
            payload.abstractive,
            payload.embeddings,
            payload.max_paragraphs,
            app.state.settings,
            app.state.processor,
        )
        return JobResponse(
            id=job_id,
            title=title,
            status="queued",
            message=message,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(created_at),
        )

    @app.get("/api/jobs/{job_id}", response_model=JobResponse)
    def get_job(
        job_id: str,
        _authorized: None = Depends(require_admin_token),
    ) -> JobResponse:
        with _connection(db_path) as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Tarefa não encontrada.")
        return JobResponse(
            id=row["id"],
            title=row["title"],
            status=row["status"],
            message=row["message"],
            error=row["error"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @app.get("/api/memories", response_model=list[MemoryResponse])
    def list_memories(
        query: str | None = Query(default=None, max_length=200),
        status: Literal["pending", "approved", "rejected"] = "approved",
        limit: int = Query(default=100, ge=1, le=300),
        _authorized: None = Depends(require_admin_token),
    ) -> list[MemoryResponse]:
        clauses = ["status = ?"]
        parameters: list[object] = [status]
        if query and query.strip():
            clauses.append("(subject LIKE ? OR fact LIKE ? OR tags_json LIKE ?)")
            needle = f"%{query.strip()}%"
            parameters.extend([needle, needle, needle])
        parameters.append(limit)
        with _connection(db_path) as connection:
            rows = connection.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(clauses)} "
                "ORDER BY created_at DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [_memory_from_row(row) for row in rows]

    @app.patch("/api/memories/{memory_id}", response_model=MemoryResponse)
    def update_memory(
        memory_id: str,
        payload: MemoryStatusUpdate,
        _authorized: None = Depends(require_admin_token),
    ) -> MemoryResponse:
        with _connection(db_path) as connection:
            cursor = connection.execute(
                "UPDATE memories SET status = ? WHERE id = ?",
                (payload.status, memory_id),
            )
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if cursor.rowcount != 1 or row is None:
            raise HTTPException(status_code=404, detail="Memória não encontrada.")
        return _memory_from_row(row)

    @app.post("/api/import", response_model=ImportResponse)
    def import_pack(
        pack: MemoryPack,
        _authorized: None = Depends(require_admin_token),
    ) -> ImportResponse:
        inserted = 0
        with _connection(db_path) as connection:
            for record in pack.memories:
                status: Literal["pending", "approved", "rejected"] = (
                    "approved" if record.approved else "pending"
                )
                if _insert_memory(
                    connection,
                    record,
                    status=status,
                    article_title=pack.title,
                    source_url=str(pack.source_url) if pack.source_url else None,
                ):
                    inserted += 1
        return ImportResponse(imported=inserted, duplicates=len(pack.memories) - inserted)

    @app.get("/api/export", response_model=MemoryPack)
    def export_pack(
        include_pending: bool = False,
        _authorized: None = Depends(require_admin_token),
    ) -> MemoryPack:
        statuses = ("pending", "approved") if include_pending else ("approved",)
        placeholders = ",".join("?" for _ in statuses)
        with _connection(db_path) as connection:
            rows = connection.execute(
                f"SELECT * FROM memories WHERE status IN ({placeholders}) ORDER BY created_at ASC",
                statuses,
            ).fetchall()
        records = [
            MemoryRecord(
                subject=row["subject"],
                fact=row["fact"],
                tags=json.loads(row["tags_json"]),
                aliases=json.loads(row["aliases_json"]),
                approved=row["status"] == "approved",
                confidence=row["confidence"],
                source=row["source"],
            )
            for row in rows
        ]
        return MemoryPack(
            title="Memórias consolidadas da Wikipédia em português",
            exported_at=datetime.now(UTC),
            source_url=None,
            memories=records,
        )

    @app.get("/api/summary")
    def summary(_authorized: None = Depends(require_admin_token)) -> dict[str, int]:
        with _connection(db_path) as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM memories GROUP BY status"
            ).fetchall()
        counts = {row["status"]: row["count"] for row in rows}
        return {
            "pending": counts.get("pending", 0),
            "approved": counts.get("approved", 0),
            "rejected": counts.get("rejected", 0),
        }

    return app


def _set_job_state(
    database_path: Path,
    job_id: str,
    *,
    status: Literal["queued", "downloading", "analyzing", "saving", "completed", "failed"],
    message: str,
    error: str | None = None,
    result_json: str | None = None,
) -> None:
    with _connection(database_path) as connection:
        connection.execute(
            "UPDATE jobs SET status = ?, message = ?, error = ?, result_json = ?, "
            "updated_at = ? WHERE id = ?",
            (status, message, error, result_json, _now(), job_id),
        )


def _run_harvest(
    database_path: Path,
    job_id: str,
    title: str,
    abstractive: bool,
    embeddings: bool,
    max_paragraphs: int,
    settings: Settings,
    processor: NLPProcessor,
) -> None:
    """Executa um harvest fora do request e persiste o resultado e a fila."""
    try:
        with _analysis_lock:
            _set_job_state(
                database_path,
                job_id,
                status="downloading",
                message="Consultando a Wikipédia em português.",
            )
            user_agent = settings.resolve_user_agent()
            article = asyncio.run(_fetch_article(settings, title, user_agent))
            _set_job_state(
                database_path,
                job_id,
                status="analyzing",
                message="Extraindo entidades, tópicos, relações e vetores semânticos.",
            )
            analysis = processor.analyze(
                article,
                abstractive=abstractive,
                embeddings=embeddings,
                max_paragraphs=max_paragraphs,
            )
            result = build_harvest_result(article, analysis, settings)
            result = HarvestResult.model_validate(result.model_dump(mode="python"))
            result_json = result.model_dump_json()
            _set_job_state(
                database_path,
                job_id,
                status="saving",
                message="Salvando memórias pendentes de revisão.",
            )
            with _connection(database_path) as connection:
                for record in result.memory_pack.memories:
                    _insert_memory(
                        connection,
                        record,
                        status="pending",
                        article_title=result.article.title,
                        source_url=str(result.article.canonical_url),
                    )
                connection.execute(
                    "UPDATE jobs SET status = 'completed', message = ?, error = NULL, "
                    "result_json = ?, updated_at = ? WHERE id = ?",
                    (
                        "Análise concluída: "
                        f"{len(result.memory_pack.memories)} memórias enviadas para revisão.",
                        result_json,
                        _now(),
                        job_id,
                    ),
                )
    except WikipediaNotFoundError as error:
        _set_job_state(
            database_path,
            job_id,
            status="failed",
            message="O título não corresponde a um artigo existente.",
            error=str(error),
        )
    except WikipediaDisambiguationError as error:
        suggestions = ", ".join(error.suggestions[:8])
        detail = f" Opções: {suggestions}." if suggestions else ""
        _set_job_state(
            database_path,
            job_id,
            status="failed",
            message="A Wikipédia encontrou uma página de desambiguação.",
            error=f"“{error.title}” é ambíguo.{detail}",
        )
    except (
        ConfigurationError,
        NLPDependencyError,
        NLPError,
        WikipediaError,
        OSError,
        ValueError,
    ) as error:
        _set_job_state(
            database_path,
            job_id,
            status="failed",
            message="Não foi possível concluir a análise.",
            error=str(error),
        )
    except Exception:
        # O detalhe técnico completo fica nos logs do servidor; ao cliente não
        # são expostos caminhos, cabeçalhos ou conteúdo interno do processo.
        import logging

        logging.getLogger(__name__).exception("Falha inesperada no job %s", job_id)
        _set_job_state(
            database_path,
            job_id,
            status="failed",
            message="Ocorreu uma falha interna durante a análise.",
            error="Tente novamente. Se o problema continuar, consulte os logs do serviço.",
        )


async def _fetch_article(settings: Settings, title: str, user_agent: str):
    async with WikipediaClient(settings, user_agent=user_agent) as client:
        return await client.fetch_article(title)


app = create_app()
