"""Configuração validada e carregada do ambiente."""

from __future__ import annotations

import re

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from wikize_memory.errors import ConfigurationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    user_agent: str = Field(default="", validation_alias="WIKIMEDIA_USER_AGENT")
    api_url: str = Field(
        default="https://pt.wikipedia.org/w/api.php",
        validation_alias="WIKIMEDIA_API_URL",
    )
    request_timeout_seconds: float = Field(default=30.0, ge=1.0, le=180.0)
    trust_env: bool = Field(default=True, validation_alias="WIKIZEMEMORY_TRUST_ENV")
    max_retries: int = Field(default=4, ge=0, le=10)
    retry_base_seconds: float = Field(default=0.8, ge=0.0, le=30.0)
    retry_max_seconds: float = Field(default=20.0, ge=0.0, le=120.0)
    max_continuations: int = Field(default=40, ge=1, le=500)
    max_article_characters: int = Field(default=1_000_000, ge=10_000)
    max_nlp_characters: int = Field(default=350_000, ge=10_000)
    spacy_model: str = Field(
        default="pt_core_news_lg",
        validation_alias="WIKIZEMEMORY_SPACY_MODEL",
    )
    summarization_model: str = Field(
        default="csebuetnlp/mT5_multilingual_XLSum",
        validation_alias="WIKIZEMEMORY_SUMMARY_MODEL",
    )
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        validation_alias="WIKIZEMEMORY_EMBEDDING_MODEL",
    )
    device: str = Field(default="auto", validation_alias="WIKIZEMEMORY_DEVICE")
    max_paragraphs: int = Field(default=8, ge=1, le=30)
    max_memory_records: int = Field(default=100, ge=1, le=1200)

    def resolve_user_agent(self, override: str | None = None) -> str:
        value = (override or self.user_agent).strip()
        if not value:
            raise ConfigurationError(
                "Configure WIKIMEDIA_USER_AGENT no arquivo .env ou use --user-agent. "
                "O valor deve identificar o bot e incluir um contato público do operador."
            )
        lowered = value.casefold()
        placeholders = (
            "seu_email",
            "seu-email",
            "seu_usuario",
            "seu-usuario",
            "seu-contato",
            "example.com",
            ".example",
        )
        if any(marker in lowered for marker in placeholders):
            raise ConfigurationError(
                "WIKIMEDIA_USER_AGENT ainda contém um exemplo. "
                "Substitua-o por um contato público real."
            )
        has_contact = bool(
            re.search(r"https?://[^\s)]+", value, flags=re.IGNORECASE)
            or re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", value)
        )
        if len(value) < 12 or not has_contact:
            raise ConfigurationError(
                "WIKIMEDIA_USER_AGENT precisa identificar o bot e conter URL ou e-mail de contato."
            )
        return value
