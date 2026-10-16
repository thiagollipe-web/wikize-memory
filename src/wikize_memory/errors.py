"""Exceções de domínio para a aplicação."""


class WikizeError(Exception):
    """Erro base que pode ser apresentado diretamente no CLI."""


class ConfigurationError(WikizeError):
    """Configuração ausente ou inválida."""


class WikipediaError(WikizeError):
    """Erro na consulta da Action API."""


class WikipediaNotFoundError(WikipediaError):
    """O título não corresponde a uma página existente."""


class WikipediaDisambiguationError(WikipediaError):
    """O título consultado é uma página de desambiguação."""

    def __init__(self, title: str, suggestions: list[str] | None = None) -> None:
        self.title = title
        self.suggestions = suggestions or []
        detail = ", ".join(self.suggestions[:8])
        suffix = f" Opções: {detail}." if detail else ""
        super().__init__(f"“{title}” é uma página de desambiguação.{suffix}")


class WikipediaProtocolError(WikipediaError):
    """Resposta inválida ou inesperada da API."""


class WikipediaPaginationError(WikipediaError):
    """O limite configurado de páginas de continuação foi atingido."""


class NLPError(WikizeError):
    """Falha no processamento linguístico."""


class NLPDependencyError(NLPError):
    """Dependência ou modelo NLP não instalado."""


class OutputError(WikizeError):
    """Falha ao validar ou gravar um artefato."""
