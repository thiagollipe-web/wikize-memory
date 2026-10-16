"""Orquestra as análises e cria a saída completa e o pacote Synapse."""

from __future__ import annotations

from datetime import UTC, datetime

from wikize_memory.config import Settings
from wikize_memory.schemas import (
    ArticleAnalysis,
    ArticleDocument,
    ArticleMetadata,
    HarvestResult,
    MemoryPack,
    MemoryRecord,
)


def build_memory_pack(
    article: ArticleDocument,
    analysis: ArticleAnalysis,
    *,
    max_records: int = 100,
) -> MemoryPack:
    """Cria fatos importáveis; todos ficam pendentes de revisão humana."""
    source = f"{article.canonical_url} | CC BY-SA 4.0"
    memories: list[MemoryRecord] = []
    summary = analysis.abstractive_summary or analysis.extractive_summary
    if summary.strip():
        memories.append(
            MemoryRecord(
                subject=article.title,
                fact=summary,
                tags=_unique(analysis.topics[:15]),
                aliases=_unique(
                    [article.requested_title]
                    if article.requested_title.casefold() != article.title.casefold()
                    else []
                ),
                approved=False,
                confidence=0.55,
                source=source,
            )
        )

    for relation in analysis.relations[: min(30, max_records)]:
        memories.append(
            MemoryRecord(
                subject=relation.subject,
                fact=f"{relation.subject} {relation.predicate} {relation.object}.",
                tags=_unique(["relação", *analysis.topics[:8]]),
                aliases=[],
                approved=False,
                confidence=relation.confidence,
                source=source,
            )
        )

    for keyphrase in analysis.keyphrases:
        phrase = keyphrase.phrase.strip()
        if not phrase:
            continue
        relevant = next(
            (
                paragraph.text
                for paragraph in analysis.paragraphs
                if phrase.casefold() in paragraph.text.casefold()
            ),
            None,
        )
        if not relevant:
            continue
        memories.append(
            MemoryRecord(
                subject=phrase,
                fact=relevant,
                tags=_unique([phrase, *analysis.topics[:8]]),
                aliases=[article.title],
                approved=False,
                confidence=0.5,
                source=source,
            )
        )
        if len(memories) >= max_records:
            break

    deduplicated: list[MemoryRecord] = []
    seen: set[tuple[str, str]] = set()
    for item in memories:
        key = (item.subject.casefold().strip(), item.fact.casefold().strip())
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
        if len(deduplicated) >= max_records:
            break

    return MemoryPack(
        title=f"Memórias da Wikipédia PT — {article.title}",
        exported_at=datetime.now(UTC),
        source_url=article.canonical_url,
        content_license="CC BY-SA 4.0",
        memories=deduplicated,
    )


def build_harvest_result(
    article: ArticleDocument,
    analysis: ArticleAnalysis,
    settings: Settings,
) -> HarvestResult:
    revision_text = f" revisão {article.revision_id}" if article.revision_id is not None else ""
    attribution = (
        f"Wikipédia, colaboradores do artigo “{article.title}”, "
        f"{article.canonical_url}{revision_text}; licença CC BY-SA 4.0 "
        "(https://creativecommons.org/licenses/by-sa/4.0/)."
    )
    metadata = ArticleMetadata(
        requested_title=article.requested_title,
        title=article.title,
        page_id=article.page_id,
        canonical_url=article.canonical_url,
        revision_id=article.revision_id,
        revision_timestamp=article.revision_timestamp,
        retrieved_at=article.retrieved_at,
        categories=article.categories,
        redirect_chain=article.redirect_chain,
        text_characters=article.original_text_characters,
        content_truncated=article.content_truncated,
        attribution=attribution,
    )
    memory_pack = build_memory_pack(
        article,
        analysis,
        max_records=settings.max_memory_records,
    )
    return HarvestResult(
        generated_at=datetime.now(UTC),
        article=metadata,
        analysis=analysis,
        memory_pack=memory_pack,
    )


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = value.strip()
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result[:30]
