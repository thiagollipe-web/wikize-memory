"""Modelos de dados estritos para artigos, análise e integração Synapse."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_default=True)


class ArticleDocument(StrictModel):
    """Documento de trabalho; o texto bruto nunca é serializado no resultado."""

    requested_title: str = Field(min_length=1, max_length=300)
    title: str = Field(min_length=1, max_length=300)
    page_id: int = Field(gt=0, strict=True)
    canonical_url: HttpUrl
    revision_id: int | None = Field(default=None, ge=0, strict=True)
    revision_timestamp: datetime | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    categories: list[str] = Field(default_factory=list, max_length=500)
    redirect_chain: list[str] = Field(default_factory=list, max_length=50)
    content: str = Field(repr=False, exclude=True)
    original_text_characters: int = Field(ge=0, strict=True)
    content_truncated: bool = False

    @field_validator("retrieved_at", "revision_timestamp")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps precisam conter fuso horário")
        return value


class ArticleMetadata(StrictModel):
    requested_title: str
    title: str
    page_id: int = Field(gt=0, strict=True)
    canonical_url: HttpUrl
    revision_id: int | None = Field(default=None, ge=0, strict=True)
    revision_timestamp: datetime | None = None
    retrieved_at: datetime
    categories: list[str] = Field(default_factory=list)
    redirect_chain: list[str] = Field(default_factory=list)
    text_characters: int = Field(ge=0, strict=True)
    content_truncated: bool
    attribution: str
    content_license: str = "CC BY-SA 4.0"
    license_url: HttpUrl = "https://creativecommons.org/licenses/by-sa/4.0/"

    @field_validator("retrieved_at", "revision_timestamp")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps precisam conter fuso horário")
        return value


class Entity(StrictModel):
    text: str = Field(min_length=1, max_length=300)
    label: str = Field(min_length=1, max_length=80)
    start_char: int | None = Field(default=None, ge=0, strict=True)
    end_char: int | None = Field(default=None, ge=0, strict=True)
    source: Literal["spacy", "domain_lexicon"] = "spacy"


class Relation(StrictModel):
    subject: str = Field(min_length=1, max_length=300)
    predicate: str = Field(min_length=1, max_length=160)
    object: str = Field(min_length=1, max_length=300)
    evidence: str = Field(min_length=1, max_length=1200)
    confidence: float = Field(ge=0.0, le=1.0)


class Keyphrase(StrictModel):
    phrase: str = Field(min_length=1, max_length=160)
    score: float = Field(ge=0.0, le=1.0)
    frequency: int = Field(ge=1, strict=True)


class ParagraphMemory(StrictModel):
    rank: int = Field(ge=1, strict=True)
    text: str = Field(min_length=1, max_length=5000)
    embedding: list[float] | None = Field(default=None, max_length=4096)

    @field_validator("embedding")
    @classmethod
    def finite_embedding(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and any(not math.isfinite(item) for item in value):
            raise ValueError("embeddings precisam conter somente valores finitos")
        return value


class ModelMetadata(StrictModel):
    language: str = "pt"
    ner_model: str
    abstractive_model: str | None
    embedding_model: str | None
    embedding_dimension: int | None = Field(default=None, ge=1, strict=True)
    abstractive_summary_available: bool
    embeddings_available: bool


class ConceptNode(StrictModel):
    id: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=300)
    kind: Literal["entity", "keyphrase", "concept"]
    occurrences: int = Field(default=1, ge=1, strict=True)


class ConceptEdge(StrictModel):
    source_id: str = Field(min_length=1, max_length=120)
    target_id: str = Field(min_length=1, max_length=120)
    relation: str = Field(min_length=1, max_length=160)
    evidence: str = Field(min_length=1, max_length=1200)
    confidence: float = Field(ge=0.0, le=1.0)


class ConceptGraph(StrictModel):
    nodes: list[ConceptNode] = Field(default_factory=list, max_length=2000)
    edges: list[ConceptEdge] = Field(default_factory=list, max_length=1000)

    @field_validator("nodes")
    @classmethod
    def unique_node_ids(cls, value: list[ConceptNode]) -> list[ConceptNode]:
        identifiers = [node.id for node in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("ids de nós do grafo precisam ser únicos")
        return value

    @field_validator("edges")
    @classmethod
    def edges_have_nodes(cls, value: list[ConceptEdge], info):
        nodes = info.data.get("nodes", [])
        identifiers = {node.id for node in nodes}
        for edge in value:
            if edge.source_id not in identifiers or edge.target_id not in identifiers:
                raise ValueError("cada extremidade de uma aresta precisa existir no grafo")
        return value


class ArticleAnalysis(StrictModel):
    extractive_summary: str = Field(min_length=1, max_length=5000)
    abstractive_summary: str | None = Field(default=None, max_length=3000)
    central_claims: list[str] = Field(default_factory=list, max_length=20)
    topics: list[str] = Field(default_factory=list, max_length=30)
    keyphrases: list[Keyphrase] = Field(default_factory=list, max_length=50)
    entities: list[Entity] = Field(default_factory=list, max_length=1000)
    relations: list[Relation] = Field(default_factory=list, max_length=1000)
    concept_graph: ConceptGraph = Field(default_factory=ConceptGraph)
    paragraphs: list[ParagraphMemory] = Field(default_factory=list, max_length=30)
    models: ModelMetadata
    warnings: list[str] = Field(default_factory=list, max_length=50)


class MemoryRecord(StrictModel):
    subject: str = Field(min_length=1, max_length=300)
    fact: str = Field(min_length=1, max_length=5000)
    tags: list[str] = Field(default_factory=list, max_length=30)
    aliases: list[str] = Field(default_factory=list, max_length=30)
    approved: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = Field(min_length=1, max_length=1000)


class MemoryPack(BaseModel):
    """Formato importável pelo THIGAS Synapse v3; aceita metadados extras na entrada."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, validate_default=True)

    format: Literal["thigas-synapse-memories"] = "thigas-synapse-memories"
    version: int = Field(default=3, ge=1, strict=True)
    title: str = "Memórias da Wikipédia em português"
    exported_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_url: HttpUrl | None = None
    content_license: str | None = "CC BY-SA 4.0"
    memories: list[MemoryRecord] = Field(default_factory=list, max_length=1200)


class HarvestResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    article: ArticleMetadata
    analysis: ArticleAnalysis
    memory_pack: MemoryPack

    @field_validator("generated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("generated_at precisa conter fuso horário")
        return value
