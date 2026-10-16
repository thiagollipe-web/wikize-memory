from __future__ import annotations

import pytest
from pydantic import ValidationError

from wikize_memory.config import Settings
from wikize_memory.errors import ConfigurationError
from wikize_memory.nlp.processor import NLPProcessor
from wikize_memory.pipeline import build_harvest_result
from wikize_memory.schemas import (
    ArticleAnalysis,
    ArticleDocument,
    Keyphrase,
    MemoryPack,
    MemoryRecord,
    ModelMetadata,
    ParagraphMemory,
    Relation,
)
from wikize_memory.utils import merge_memory_packs


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        MemoryRecord(
            subject="Memória",
            fact="Fato",
            confidence=1.2,
            source="https://pt.wikipedia.org/wiki/Mem%C3%B3ria",
        )


def test_merge_deduplicates_and_preserves_review_state():
    existing = MemoryPack(
        title="Biblioteca",
        memories=[
            MemoryRecord(
                subject="Memória",
                fact="A memória permite guardar informações.",
                tags=["cognição"],
                aliases=["lembrança"],
                approved=True,
                confidence=0.9,
                source="biblioteca-local",
            )
        ],
    )
    incoming = MemoryPack(
        title="Novo artigo",
        memories=[
            MemoryRecord(
                subject="memoria",
                fact="A memória permite guardar informações.",
                tags=["psicologia"],
                aliases=["recordação"],
                approved=False,
                confidence=0.5,
                source="https://pt.wikipedia.org/wiki/Mem%C3%B3ria",
            )
        ],
    )
    merged = merge_memory_packs(existing, incoming)

    assert len(merged.memories) == 1
    assert merged.memories[0].approved is True
    assert merged.memories[0].confidence == 0.9
    assert merged.memories[0].tags == ["cognição", "psicologia"]
    assert merged.memories[0].aliases == ["lembrança", "recordação"]


def test_harvest_result_marks_generated_memories_for_review():
    settings = Settings(_env_file=None, user_agent="WikiBot/1.0 (+https://example.org)")
    article = ArticleDocument(
        requested_title="Memória",
        title="Memória",
        page_id=123,
        canonical_url="https://pt.wikipedia.org/wiki/Mem%C3%B3ria",
        revision_id=456,
        content=(
            "A memória participa da aprendizagem e da cognição. O hipocampo ajuda na consolidação."
        ),
        original_text_characters=90,
        categories=["Cognição"],
    )
    analysis = ArticleAnalysis(
        extractive_summary="A memória participa da aprendizagem e da cognição.",
        central_claims=["A memória participa da aprendizagem e da cognição."],
        topics=["memória", "aprendizagem"],
        keyphrases=[Keyphrase(phrase="memória", score=1.0, frequency=2)],
        paragraphs=[
            ParagraphMemory(
                rank=1,
                text="A memória participa da aprendizagem e da cognição.",
                embedding=[0.1, 0.2],
            )
        ],
        models=ModelMetadata(
            ner_model="pt_core_news_lg",
            abstractive_model=None,
            embedding_model="test-model",
            embedding_dimension=2,
            abstractive_summary_available=False,
            embeddings_available=True,
        ),
    )
    result = build_harvest_result(article, analysis, settings)

    assert result.article.content_license == "CC BY-SA 4.0"
    assert result.memory_pack.memories
    assert all(item.approved is False for item in result.memory_pack.memories)
    assert all("wikipedia.org" in item.source for item in result.memory_pack.memories)


def test_user_agent_rejects_placeholder_contact():
    settings = Settings(_env_file=None, user_agent="")
    with pytest.raises(ConfigurationError, match="exemplo"):
        settings.resolve_user_agent("WikiMemory/1.0 (+https://seu-contato-publico.example)")


def test_concept_graph_has_nodes_and_valid_relation_edges():
    graph = NLPProcessor._build_concept_graph(
        [],
        [Keyphrase(phrase="memória de trabalho", score=1.0, frequency=2)],
        [
            Relation(
                subject="hipocampo",
                predicate="participar",
                object="consolidação",
                evidence="O hipocampo participa da consolidação.",
                confidence=0.55,
            )
        ],
    )
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 1
    assert graph.edges[0].source_id in {node.id for node in graph.nodes}
    assert graph.edges[0].target_id in {node.id for node in graph.nodes}
