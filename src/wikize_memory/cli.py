"""Interface Typer/Rich."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from wikize_memory import __version__
from wikize_memory.config import Settings
from wikize_memory.errors import WikizeError
from wikize_memory.ingestion import WikipediaClient
from wikize_memory.nlp import NLPProcessor
from wikize_memory.pipeline import build_harvest_result
from wikize_memory.schemas import HarvestResult
from wikize_memory.utils import load_memory_pack, merge_memory_packs, write_json_atomic

app = typer.Typer(
    name="wikize",
    help="Extrai, analisa e consolida memórias de artigos da Wikipédia PT.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def harvest(
    title: str = typer.Argument(..., help="Título de artigo da Wikipédia em português."),
    output: Path = typer.Option(
        Path("data/memoria-wikipedia.json"),
        "--output",
        "-o",
        help="Caminho do JSON completo com análise e memória.",
    ),
    synapse_output: Path | None = typer.Option(
        None,
        "--synapse-output",
        help="Gera também um JSON importável pelo THIGAS Synapse v3.",
    ),
    merge_existing: Path | None = typer.Option(
        None,
        "--merge-existing",
        help="Mescla o pacote novo com um JSON de memórias já existente.",
    ),
    user_agent: str | None = typer.Option(
        None,
        "--user-agent",
        help="User-Agent identificável e com contato público do operador.",
    ),
    no_abstractive: bool = typer.Option(
        False,
        "--no-abstractive",
        help="Pula o modelo mT5; mantém o resumo extrativo.",
    ),
    no_embeddings: bool = typer.Option(
        False,
        "--no-embeddings",
        help="Pula embeddings para reduzir memória e tempo de execução.",
    ),
    max_paragraphs: int = typer.Option(
        8,
        "--max-paragraphs",
        min=1,
        max=30,
        help="Número máximo de parágrafos selecionados para vetorização.",
    ),
) -> None:
    """Coleta e processa um artigo."""
    settings = Settings()
    try:
        resolved_user_agent = settings.resolve_user_agent(user_agent)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Consultando Wikipédia", total=3)
            article = asyncio.run(_fetch_article(settings, title, resolved_user_agent))
            progress.update(task, advance=1, description="Executando PNL em português")
            processor = NLPProcessor(settings)
            analysis = processor.analyze(
                article,
                abstractive=not no_abstractive,
                embeddings=not no_embeddings,
                max_paragraphs=max_paragraphs,
            )
            progress.update(task, advance=1, description="Validando saída Pydantic v2")
            result = build_harvest_result(article, analysis, settings)
            result = HarvestResult.model_validate(result.model_dump(mode="python"))
            progress.update(task, advance=1, description="Artefatos prontos")
        write_json_atomic(output, result)
        console.print(
            f"[green]JSON completo salvo:[/green] {output} "
            f"([cyan]{len(analysis.entities)}[/cyan] entidades, "
            f"[cyan]{len(analysis.relations)}[/cyan] relações, "
            f"[cyan]{len(result.memory_pack.memories)}[/cyan] memórias)"
        )

        if synapse_output or merge_existing:
            pack = result.memory_pack
            if merge_existing:
                prior = load_memory_pack(merge_existing)
                pack = merge_memory_packs(prior, pack)
            target = synapse_output or output.with_name(f"{output.stem}.synapse.json")
            write_json_atomic(target, pack)
            console.print(f"[green]Pacote THIGAS Synapse salvo:[/green] {target}")
        if analysis.warnings:
            for warning in analysis.warnings:
                console.print(f"[yellow]Aviso:[/yellow] {warning}")
        console.print(
            f"[dim]Fonte: {article.canonical_url} · CC BY-SA 4.0 · "
            f"wikize-memory {__version__}[/dim]"
        )
    except (WikizeError, ValidationError, OSError, ValueError) as error:
        console.print(f"[bold red]Erro:[/bold red] {error}")
        raise typer.Exit(code=1) from error


@app.command()
def validate(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Arquivo JSON para validar."),
) -> None:
    """Valida um artefato completo ou pacote de memórias Synapse."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "schema_version" in payload:
            parsed = HarvestResult.model_validate(payload)
            console.print(
                f"[green]Artefato válido:[/green] {parsed.article.title} "
                f"({len(parsed.analysis.entities)} entidades)"
            )
        else:
            pack = load_memory_pack(path)
            console.print(
                f"[green]Pacote Synapse válido:[/green] "
                f"{len(pack.memories)} memórias · versão {pack.version}"
            )
    except (WikizeError, ValidationError, OSError, json.JSONDecodeError) as error:
        console.print(f"[bold red]JSON inválido:[/bold red] {error}")
        raise typer.Exit(code=1) from error


@app.command()
def version() -> None:
    """Mostra a versão instalada."""
    console.print(f"wikize-memory {__version__}")


async def _fetch_article(settings: Settings, title: str, user_agent: str):
    async with WikipediaClient(settings, user_agent=user_agent) as client:
        return await client.fetch_article(title)
