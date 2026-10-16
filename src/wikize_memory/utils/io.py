"""Leitura, validação, consolidação e gravação atômica de JSON."""

from __future__ import annotations

import json
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from wikize_memory.errors import OutputError
from wikize_memory.schemas import MemoryPack, MemoryRecord


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    temporary_path: Path | None = None
    try:
        encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    except (OSError, TypeError, ValueError) as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise OutputError(f"Não foi possível gravar {path}: {error}") from error


def load_memory_pack(path: Path) -> MemoryPack:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OutputError(f"Não foi possível ler o JSON {path}: {error}") from error
    if isinstance(payload, list):
        payload = {"memories": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("memories"), list):
        raise OutputError(
            "O arquivo precisa ser uma lista de memórias ou um objeto com campo memories."
        )
    try:
        return MemoryPack.model_validate(payload)
    except ValidationError as error:
        raise OutputError(f"Pacote de memórias inválido: {error}") from error


def merge_memory_packs(existing: MemoryPack, incoming: MemoryPack) -> MemoryPack:
    """Mescla por assunto + fato normalizados, preservando aprovações existentes."""
    merged: list[MemoryRecord] = list(existing.memories)
    indices = {_memory_key(item): index for index, item in enumerate(merged)}
    for item in incoming.memories:
        key = _memory_key(item)
        if key not in indices:
            merged.append(item)
            indices[key] = len(merged) - 1
            continue

        index = indices[key]
        prior = merged[index]
        merged[index] = prior.model_copy(
            update={
                "tags": _merge_strings(prior.tags, item.tags),
                "aliases": _merge_strings(prior.aliases, item.aliases),
                "approved": prior.approved or item.approved,
                "confidence": max(prior.confidence, item.confidence),
            }
        )

    if len(merged) > 1200:
        raise OutputError(
            f"A consolidação geraria {len(merged)} memórias; o limite do Synapse é 1.200."
        )
    return MemoryPack(
        title=existing.title or incoming.title,
        exported_at=incoming.exported_at,
        source_url=incoming.source_url or existing.source_url,
        content_license=incoming.content_license or existing.content_license,
        memories=merged,
    )


def _memory_key(item: MemoryRecord) -> tuple[str, str]:
    return (_fold(item.subject), _fold(item.fact))


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char)).strip()


def _merge_strings(first: list[str], second: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in [*first, *second]:
        clean = value.strip()
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result[:30]
