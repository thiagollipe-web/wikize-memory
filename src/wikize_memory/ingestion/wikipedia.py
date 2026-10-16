"""Cliente assíncrono e resiliente da MediaWiki Action API."""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from wikize_memory.config import Settings
from wikize_memory.errors import (
    WikipediaDisambiguationError,
    WikipediaError,
    WikipediaNotFoundError,
    WikipediaPaginationError,
    WikipediaProtocolError,
)
from wikize_memory.schemas import ArticleDocument

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_RETRYABLE_API_CODES = {"maxlag", "ratelimited", "readonly"}


class WikipediaClient:
    """Obtém um verbete, suas categorias, links e dados de revisão."""

    def __init__(
        self,
        settings: Settings,
        *,
        user_agent: str | None = None,
        sleep=asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.user_agent = settings.resolve_user_agent(user_agent)
        self._sleep = sleep
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_seconds),
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
                "Accept-Language": "pt-BR,pt;q=0.9",
            },
            follow_redirects=True,
            trust_env=settings.trust_env,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )

    async def __aenter__(self) -> WikipediaClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch_article(self, title: str) -> ArticleDocument:
        requested_title = title.strip()
        if not requested_title:
            raise WikipediaNotFoundError("O título não pode ficar vazio.")

        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "maxlag": 5,
            "redirects": 1,
            "converttitles": 1,
            "titles": requested_title,
            "prop": "extracts|info|categories|links|pageprops|revisions",
            "inprop": "url",
            "explaintext": 1,
            "exsectionformat": "plain",
            "cllimit": "max",
            "pllimit": 20,
            "rvprop": "ids|timestamp",
        }
        first_page: dict[str, Any] | None = None
        categories: list[str] = []
        redirect_chain: list[str] = []
        continuation_count = 0

        while True:
            payload = await self._request_json(params)
            query = payload.get("query")
            if not isinstance(query, dict):
                raise WikipediaProtocolError("A resposta da API não contém o objeto query.")

            for redirect in query.get("redirects", []) or []:
                target = redirect.get("to")
                if isinstance(target, str) and target not in redirect_chain:
                    redirect_chain.append(target)

            pages = query.get("pages")
            if not isinstance(pages, list) or not pages:
                raise WikipediaProtocolError("A API retornou uma lista de páginas vazia.")
            page = pages[0]
            if not isinstance(page, dict):
                raise WikipediaProtocolError("A API retornou uma página inválida.")

            if first_page is None:
                first_page = page
                if "missing" in page:
                    raise WikipediaNotFoundError(
                        f"A Wikipédia não encontrou uma página para “{requested_title}”."
                    )
                pageprops = page.get("pageprops", {}) or {}
                if "disambiguation" in pageprops:
                    choices = [
                        item.get("title", "")
                        for item in page.get("links", []) or []
                        if item.get("title")
                    ]
                    raise WikipediaDisambiguationError(requested_title, choices)

            categories.extend(
                item.get("title", "")
                for item in page.get("categories", []) or []
                if item.get("title")
            )
            continuation = payload.get("continue")
            if not isinstance(continuation, dict) or "clcontinue" not in continuation:
                break
            continuation_count += 1
            if continuation_count >= self.settings.max_continuations:
                raise WikipediaPaginationError(
                    "O artigo excedeu o limite de páginas de continuação. "
                    "Aumente max_continuations se precisar coletar mais categorias e links."
                )
            params.update(continuation)

        assert first_page is not None
        raw_text = str(first_page.get("extract", "")).strip()
        if not raw_text:
            raise WikipediaProtocolError(
                f"A página “{first_page.get('title', requested_title)}” não contém texto extraível."
            )

        revisions = first_page.get("revisions", []) or []
        revision = revisions[0] if revisions and isinstance(revisions[0], dict) else {}
        timestamp = revision.get("timestamp")
        parsed_timestamp: datetime | None = None
        if timestamp:
            try:
                parsed_timestamp = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            except ValueError as error:
                raise WikipediaProtocolError("Timestamp de revisão inválido.") from error

        full_url = first_page.get("fullurl")
        if not full_url:
            raise WikipediaProtocolError("A resposta não contém a URL canônica do artigo.")

        truncated_text = raw_text[: self.settings.max_article_characters]
        clean_categories = sorted(
            {item.removeprefix("Categoria:").strip() for item in categories if item.strip()},
            key=str.casefold,
        )
        return ArticleDocument(
            requested_title=requested_title,
            title=str(first_page.get("title", requested_title)),
            page_id=int(first_page["pageid"]),
            canonical_url=full_url,
            revision_id=int(revision["revid"]) if revision.get("revid") is not None else None,
            revision_timestamp=parsed_timestamp,
            retrieved_at=datetime.now(UTC),
            categories=clean_categories,
            redirect_chain=redirect_chain,
            content=truncated_text,
            original_text_characters=len(raw_text),
            content_truncated=len(raw_text) > len(truncated_text),
        )

    async def _request_json(self, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.settings.max_retries + 1):
            retry_delay: float | None = None
            try:
                response = await self._client.get(self.settings.api_url, params=params)
                if response.status_code in _RETRYABLE_STATUS:
                    if attempt >= self.settings.max_retries:
                        raise WikipediaError(
                            f"A Wikipédia respondeu HTTP {response.status_code} após "
                            f"{attempt + 1} tentativas."
                        )
                    retry_delay = self._retry_delay(attempt, response.headers.get("Retry-After"))
                    await self._sleep(retry_delay)
                    continue
                if response.is_redirect:
                    location = response.headers.get("Location", "")
                    raise WikipediaError(f"Redirecionamento HTTP não resolvido: {location}")
                response.raise_for_status()
                try:
                    payload = response.json()
                except ValueError as error:
                    raise WikipediaProtocolError("A API retornou JSON inválido.") from error
                if not isinstance(payload, dict):
                    raise WikipediaProtocolError("A raiz da resposta da API não é um objeto JSON.")

                api_error = payload.get("error")
                if isinstance(api_error, dict):
                    code = str(api_error.get("code", ""))
                    message = str(api_error.get("info", code or "erro da API"))
                    if code in _RETRYABLE_API_CODES and attempt < self.settings.max_retries:
                        await self._sleep(self._retry_delay(attempt, None))
                        continue
                    raise WikipediaError(f"Erro da Action API ({code}): {message}")
                return payload
            except httpx.TooManyRedirects as error:
                raise WikipediaError("A API excedeu o limite de redirecionamentos HTTP.") from error
            except httpx.HTTPStatusError as error:
                raise WikipediaError(
                    f"Resposta HTTP não recuperável: {error.response.status_code}."
                ) from error
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                if attempt >= self.settings.max_retries:
                    raise WikipediaError(
                        f"Falha de rede após {attempt + 1} tentativas: {error.__class__.__name__}."
                    ) from error
                await self._sleep(self._retry_delay(attempt, None))

        raise WikipediaError("A consulta falhou sem retornar uma resposta.")

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                seconds = float(retry_after)
            except ValueError:
                try:
                    when = parsedate_to_datetime(retry_after)
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=UTC)
                    seconds = max(0.0, (when - datetime.now(UTC)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    seconds = self.settings.retry_base_seconds * (2**attempt)
        else:
            seconds = self.settings.retry_base_seconds * (2**attempt)
        capped = min(seconds, self.settings.retry_max_seconds)
        return capped + (random.uniform(0.0, min(0.25, capped * 0.2)) if capped > 0 else 0.0)
