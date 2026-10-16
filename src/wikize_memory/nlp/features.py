"""Funções determinísticas para frases, parágrafos e saliência extrativa."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

PORTUGUESE_STOPWORDS = {
    "a",
    "à",
    "agora",
    "aí",
    "algo",
    "algumas",
    "alguns",
    "ali",
    "além",
    "ambas",
    "ambos",
    "antes",
    "ao",
    "aos",
    "apenas",
    "após",
    "aquela",
    "aquelas",
    "aquele",
    "aqueles",
    "aqui",
    "as",
    "assim",
    "até",
    "bem",
    "cada",
    "catorze",
    "cedo",
    "cento",
    "certamente",
    "cinco",
    "coisa",
    "com",
    "como",
    "conforme",
    "conosco",
    "contudo",
    "custa",
    "da",
    "dão",
    "daquela",
    "daquelas",
    "daquele",
    "daqueles",
    "dar",
    "das",
    "de",
    "dela",
    "delas",
    "dele",
    "deles",
    "demais",
    "dentro",
    "depois",
    "desde",
    "dessa",
    "dessas",
    "desse",
    "desses",
    "desta",
    "destas",
    "deste",
    "destes",
    "deve",
    "devem",
    "devendo",
    "dever",
    "dez",
    "dezasseis",
    "dezassete",
    "dezoito",
    "dia",
    "diante",
    "disso",
    "disto",
    "dito",
    "diz",
    "dizem",
    "do",
    "dos",
    "doze",
    "duas",
    "durante",
    "e",
    "é",
    "ela",
    "elas",
    "ele",
    "eles",
    "em",
    "embora",
    "entre",
    "era",
    "eram",
    "essa",
    "essas",
    "esse",
    "esses",
    "esta",
    "está",
    "estão",
    "estas",
    "estava",
    "estavam",
    "este",
    "esteja",
    "estejam",
    "estejamos",
    "estes",
    "esteve",
    "estive",
    "estivemos",
    "estiver",
    "estivera",
    "estiveram",
    "estiverem",
    "estivermos",
    "estivesse",
    "estivessem",
    "estivéssemos",
    "estou",
    "estás",
    "estávamos",
    "estiveste",
    "estivestes",
    "eu",
    "faz",
    "fazeis",
    "fazem",
    "fazemos",
    "fazendo",
    "fazer",
    "fazes",
    "feita",
    "feitas",
    "feito",
    "feitos",
    "fez",
    "fim",
    "foi",
    "fomos",
    "for",
    "fora",
    "foram",
    "forem",
    "forma",
    "formos",
    "fosse",
    "fossem",
    "fôssemos",
    "fui",
    "geral",
    "grande",
    "grandes",
    "há",
    "haja",
    "hajam",
    "hajamos",
    "hão",
    "havemos",
    "havia",
    "hei",
    "hoje",
    "houve",
    "houvemos",
    "houver",
    "houverá",
    "houveram",
    "houverão",
    "houverei",
    "houveria",
    "houveriam",
    "houvermos",
    "houvesse",
    "houvessem",
    "houvéssemos",
    "isso",
    "isto",
    "já",
    "la",
    "lá",
    "lhe",
    "lhes",
    "lo",
    "mais",
    "mas",
    "me",
    "mesma",
    "mesmas",
    "mesmo",
    "mesmos",
    "meu",
    "meus",
    "mil",
    "minha",
    "minhas",
    "menos",
    "muita",
    "muitas",
    "muito",
    "muitos",
    "na",
    "nada",
    "naquela",
    "naquelas",
    "naquele",
    "naqueles",
    "nas",
    "nem",
    "nenhum",
    "nessa",
    "nessas",
    "nesse",
    "nesses",
    "nesta",
    "nestas",
    "neste",
    "nestes",
    "ninguém",
    "no",
    "nos",
    "nós",
    "nossa",
    "nossas",
    "nosso",
    "nossos",
    "num",
    "numa",
    "nunca",
    "o",
    "obra",
    "obrigada",
    "obrigado",
    "oitava",
    "oitavo",
    "oitenta",
    "oito",
    "onde",
    "ontem",
    "onze",
    "ora",
    "os",
    "ou",
    "outra",
    "outras",
    "outro",
    "outros",
    "para",
    "parece",
    "parte",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "per",
    "perante",
    "pode",
    "podem",
    "podendo",
    "poder",
    "poderá",
    "poderão",
    "poderia",
    "poderiam",
    "podia",
    "podiam",
    "pôde",
    "pois",
    "por",
    "porque",
    "portanto",
    "posição",
    "possivelmente",
    "pouca",
    "poucas",
    "pouco",
    "poucos",
    "primeira",
    "primeiras",
    "primeiro",
    "primeiros",
    "própria",
    "próprias",
    "próprio",
    "próprios",
    "quais",
    "qual",
    "quando",
    "quanto",
    "quantos",
    "quarta",
    "quarto",
    "quatro",
    "que",
    "quem",
    "quer",
    "quereis",
    "querem",
    "queremos",
    "querer",
    "queres",
    "quero",
    "questão",
    "quinta",
    "quinto",
    "quinze",
    "são",
    "se",
    "segunda",
    "segundo",
    "seis",
    "seja",
    "sejam",
    "sejamos",
    "sem",
    "sempre",
    "sendo",
    "ser",
    "será",
    "serão",
    "serei",
    "seremos",
    "seria",
    "seriam",
    "sermos",
    "seu",
    "seus",
    "sexta",
    "sexto",
    "sim",
    "sistema",
    "sob",
    "sobre",
    "somos",
    "sou",
    "sua",
    "suas",
    "tal",
    "talvez",
    "também",
    "tampouco",
    "te",
    "tem",
    "têm",
    "temos",
    "tendes",
    "tenha",
    "tenham",
    "tenhamos",
    "tenho",
    "tens",
    "ter",
    "terá",
    "terão",
    "terei",
    "teremos",
    "teria",
    "teriam",
    "termos",
    "teu",
    "teus",
    "teve",
    "tinha",
    "tinham",
    "tive",
    "tivemos",
    "tiver",
    "tivera",
    "tiveram",
    "tiverem",
    "tivermos",
    "tivesse",
    "tivessem",
    "tivéssemos",
    "toda",
    "todas",
    "todo",
    "todos",
    "três",
    "treze",
    "tu",
    "tua",
    "tuas",
    "tudo",
    "última",
    "últimas",
    "último",
    "últimos",
    "um",
    "uma",
    "umas",
    "uns",
    "vai",
    "vais",
    "vamos",
    "vão",
    "vários",
    "várias",
    "vê",
    "vez",
    "vezes",
    "vindo",
    "vinte",
    "você",
    "vocês",
    "vos",
    "vós",
    "zero",
}
_WORD_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ0-9“(])")


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in folded if not unicodedata.combining(char))


NORMALIZED_STOPWORDS = frozenset(normalize(word) for word in PORTUGUESE_STOPWORDS)


def content_words(text: str, extra_stopwords: Iterable[str] = ()) -> list[str]:
    stop = NORMALIZED_STOPWORDS | {normalize(word) for word in extra_stopwords}
    return [
        normalize(match.group(0))
        for match in _WORD_RE.finditer(text)
        if normalize(match.group(0)) not in stop
        and len(match.group(0)) > 2
        and not match.group(0).isdigit()
    ]


def split_sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    return [part.strip() for part in _SENTENCE_RE.split(clean) if part.strip()]


def split_paragraphs(text: str) -> list[str]:
    cleaned_lines = []
    for line in text.replace("\r", "").splitlines():
        value = line.strip()
        if not value:
            cleaned_lines.append("")
            continue
        if value.startswith("==") and value.endswith("=="):
            cleaned_lines.append("")
            continue
        cleaned_lines.append(value)
    blocks = re.split(r"\n\s*\n", "\n".join(cleaned_lines))
    paragraphs = []
    for block in blocks:
        value = re.sub(r"\s+", " ", block).strip()
        if len(value) >= 60:
            paragraphs.append(value)
    if paragraphs:
        return paragraphs
    return [sentence for sentence in split_sentences(text) if len(sentence) >= 60]


def sentence_scores(sentences: list[str]) -> list[float]:
    """Pontua frases pela frequência normalizada de termos informativos do documento."""
    if not sentences:
        return []
    token_lists = [content_words(sentence) for sentence in sentences]
    frequencies = Counter(token for tokens in token_lists for token in tokens)
    if not frequencies:
        return [0.0] * len(sentences)
    raw_scores = []
    for tokens in token_lists:
        if not tokens:
            raw_scores.append(0.0)
            continue
        weighted = sum(math.log1p(frequencies[token]) for token in tokens)
        raw_scores.append(weighted / math.sqrt(len(tokens)))
    maximum = max(raw_scores, default=0.0)
    if maximum <= 0:
        return [0.0] * len(raw_scores)
    return [score / maximum for score in raw_scores]


def extractive_summary(sentences: list[str], limit: int = 5) -> tuple[str, list[str]]:
    if not sentences:
        return "", []
    scores = sentence_scores(sentences)
    selected_indices = sorted(
        sorted(range(len(sentences)), key=lambda index: scores[index], reverse=True)[:limit]
    )
    selected = [sentences[index] for index in selected_indices]
    summary = " ".join(selected).strip()
    return summary, selected


def paragraph_relevance(paragraph: str, sentences: list[str], scores: list[float]) -> float:
    paragraph_words = set(content_words(paragraph))
    related = [
        score
        for sentence, score in zip(sentences, scores, strict=False)
        if paragraph_words.intersection(content_words(sentence))
    ]
    return sum(related) / len(related) if related else 0.0
