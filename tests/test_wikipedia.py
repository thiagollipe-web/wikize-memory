from __future__ import annotations

import httpx
import pytest
import respx

from wikize_memory.errors import (
    WikipediaDisambiguationError,
    WikipediaError,
    WikipediaNotFoundError,
)
from wikize_memory.ingestion import WikipediaClient

API_URL = "https://pt.wikipedia.org/w/api.php"


def article_page(**overrides):
    page = {
        "pageid": 123,
        "title": "Memória",
        "fullurl": "https://pt.wikipedia.org/wiki/Mem%C3%B3ria",
        "extract": (
            "A memória é a capacidade de adquirir, armazenar e recuperar informações. "
            "Ela participa da aprendizagem e da cognição."
        ),
        "categories": [{"title": "Categoria:Cognição"}],
        "links": [{"title": "Cérebro"}],
        "pageprops": {},
        "revisions": [{"revid": 456, "timestamp": "2026-10-16T12:00:00Z"}],
    }
    page.update(overrides)
    return page


@pytest.mark.asyncio
async def test_fetches_page_and_follows_api_continuation(settings):
    first = {
        "query": {"pages": [article_page()]},
        "continue": {"continue": "||", "clcontinue": "next"},
    }
    second_page = article_page(
        categories=[{"title": "Categoria:Neuropsicologia"}],
        links=[{"title": "Hipocampo"}],
    )
    second = {"query": {"pages": [second_page]}}
    with respx.mock(assert_all_called=True) as router:
        router.get(API_URL).mock(
            side_effect=[
                httpx.Response(200, json=first),
                httpx.Response(200, json=second),
            ]
        )
        async with WikipediaClient(settings) as client:
            article = await client.fetch_article("Memória")

    assert article.title == "Memória"
    assert article.page_id == 123
    assert article.revision_id == 456
    assert article.revision_timestamp is not None
    assert article.categories == ["Cognição", "Neuropsicologia"]
    assert article.content.startswith("A memória")
    assert article.original_text_characters == len(article.content)


@pytest.mark.asyncio
async def test_follows_a_mediawiki_title_redirect(settings):
    payload = {
        "query": {
            "redirects": [{"from": "memoria", "to": "Memória"}],
            "pages": [article_page()],
        }
    }
    with respx.mock(assert_all_called=True) as router:
        router.get(API_URL).mock(return_value=httpx.Response(200, json=payload))
        async with WikipediaClient(settings) as client:
            article = await client.fetch_article("memoria")

    assert article.requested_title == "memoria"
    assert article.title == "Memória"
    assert article.redirect_chain == ["Memória"]


@pytest.mark.asyncio
async def test_reports_missing_page(settings):
    payload = {"query": {"pages": [{"ns": 0, "title": "NaoExiste", "missing": True}]}}
    with respx.mock(assert_all_called=True) as router:
        router.get(API_URL).mock(return_value=httpx.Response(200, json=payload))
        async with WikipediaClient(settings) as client:
            with pytest.raises(WikipediaNotFoundError):
                await client.fetch_article("NaoExiste")


@pytest.mark.asyncio
async def test_reports_disambiguation_page_with_choices(settings):
    payload = {
        "query": {
            "pages": [
                article_page(
                    pageprops={"disambiguation": ""},
                    links=[{"title": "Memória (psicologia)"}, {"title": "Memória (computação)"}],
                )
            ]
        }
    }
    with respx.mock(assert_all_called=True) as router:
        router.get(API_URL).mock(return_value=httpx.Response(200, json=payload))
        async with WikipediaClient(settings) as client:
            with pytest.raises(WikipediaDisambiguationError) as error:
                await client.fetch_article("Memória")

    assert "Memória (psicologia)" in error.value.suggestions


@pytest.mark.asyncio
async def test_retries_transient_http_status_with_backoff(settings):
    payload = {"query": {"pages": [article_page()]}}
    with respx.mock(assert_all_called=True) as router:
        route = router.get(API_URL).mock(
            side_effect=[
                httpx.Response(503, headers={"Retry-After": "0"}),
                httpx.Response(200, json=payload),
            ]
        )
        async with WikipediaClient(settings) as client:
            article = await client.fetch_article("Memória")

    assert route.call_count == 2
    assert article.page_id == 123


@pytest.mark.asyncio
async def test_raises_after_retry_budget_is_exhausted(settings):
    settings.max_retries = 0
    with respx.mock(assert_all_called=True) as router:
        router.get(API_URL).mock(return_value=httpx.Response(503))
        async with WikipediaClient(settings) as client:
            with pytest.raises(WikipediaError, match="HTTP 503"):
                await client.fetch_article("Memória")
