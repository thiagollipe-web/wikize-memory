"""Extração semântica por spaCy e sumarização/embeddings opcionais por Transformers."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

from wikize_memory.config import Settings
from wikize_memory.errors import NLPDependencyError, NLPError
from wikize_memory.nlp.features import (
    NORMALIZED_STOPWORDS,
    content_words,
    extractive_summary,
    normalize,
    paragraph_relevance,
    sentence_scores,
    split_paragraphs,
    split_sentences,
)
from wikize_memory.schemas import (
    ArticleAnalysis,
    ArticleDocument,
    ConceptEdge,
    ConceptGraph,
    ConceptNode,
    Entity,
    Keyphrase,
    ModelMetadata,
    ParagraphMemory,
    Relation,
)

DOMAIN_CONCEPTS = (
    "memória de trabalho",
    "memória episódica",
    "memória semântica",
    "memória declarativa",
    "memória procedural",
    "memória de curto prazo",
    "memória de longo prazo",
    "memória sensorial",
    "consolidação da memória",
    "evocação",
    "cognição",
    "neuropsicologia",
    "hipocampo",
    "córtex pré-frontal",
    "córtex entorrinal",
    "lobo temporal medial",
    "amígdala",
    "gânglios basais",
    "cerebelo",
    "aprendizagem motora",
    "atenção",
    "percepção",
    "processos mentais",
)
_DOMAIN_PATTERNS = [
    (concept, re.compile(r"(?<!\w)" + re.escape(concept) + r"(?!\w)", re.IGNORECASE))
    for concept in DOMAIN_CONCEPTS
]
_SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
_OBJECT_DEPS = {"obj", "dobj", "attr", "oprd", "iobj", "pobj"}


class NLPProcessor:
    """Carrega cada modelo no máximo uma vez por processo."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._nlp: Any = None
        self._summarizer: Any = None
        self._embedder: Any = None
        self._embedding_dimension: int | None = None
        self._device: str | None = None

    def analyze(
        self,
        article: ArticleDocument,
        *,
        abstractive: bool = True,
        embeddings: bool = True,
        max_paragraphs: int | None = None,
    ) -> ArticleAnalysis:
        if not article.content.strip():
            raise NLPError("O artigo está sem conteúdo para análise.")

        text = article.content[: self.settings.max_nlp_characters]
        doc = self._load_spacy()(text)
        sentences = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
        if not sentences:
            sentences = split_sentences(text)
        extractive, central_claims = extractive_summary(sentences, limit=5)
        if not extractive:
            extractive = text[:1200].strip()
        sentence_rank = sentence_scores(sentences)
        keyphrases = self._extract_keyphrases(doc)
        entities = self._extract_entities(doc)
        relations = self._extract_relations(doc)
        paragraphs = self._rank_paragraphs(text, sentences, sentence_rank)
        paragraph_limit = max_paragraphs or self.settings.max_paragraphs
        paragraphs = paragraphs[:paragraph_limit]

        warnings: list[str] = []
        abstract_summary: str | None = None
        if abstractive:
            try:
                abstract_summary = self._summarize(text)
            except Exception as error:
                warnings.append(
                    "Resumo abstrativo indisponível: "
                    f"{error.__class__.__name__}. O resumo extrativo foi preservado."
                )

        embedding_model: str | None = None
        if embeddings and paragraphs:
            try:
                vectors = self._embed([item.text for item in paragraphs])
                paragraphs = [
                    item.model_copy(update={"embedding": vector})
                    for item, vector in zip(paragraphs, vectors, strict=True)
                ]
                embedding_model = self.settings.embedding_model
            except Exception as error:
                warnings.append(
                    "Embeddings indisponíveis: "
                    f"{error.__class__.__name__}. Os parágrafos continuam exportáveis."
                )

        topics = [item.phrase for item in keyphrases[:12]]
        return ArticleAnalysis(
            extractive_summary=extractive,
            abstractive_summary=abstract_summary,
            central_claims=central_claims[:10],
            topics=topics,
            keyphrases=keyphrases,
            entities=entities,
            relations=relations,
            concept_graph=self._build_concept_graph(entities, keyphrases, relations),
            paragraphs=paragraphs,
            models=ModelMetadata(
                ner_model=self.settings.spacy_model,
                abstractive_model=self.settings.summarization_model if abstract_summary else None,
                embedding_model=embedding_model,
                embedding_dimension=self._embedding_dimension if embedding_model else None,
                abstractive_summary_available=abstract_summary is not None,
                embeddings_available=embedding_model is not None,
            ),
            warnings=warnings,
        )

    def _load_spacy(self) -> Any:
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy
        except ImportError as error:
            raise NLPDependencyError(
                "spaCy não está instalado. Instale o pacote com: pip install -e '.[nlp]'."
            ) from error
        try:
            self._nlp = spacy.load(self.settings.spacy_model)
        except OSError as error:
            raise NLPDependencyError(
                f"Modelo spaCy {self.settings.spacy_model!r} não encontrado. "
                "Instale-o com: python -m spacy download pt_core_news_lg"
            ) from error
        return self._nlp

    def _extract_entities(self, doc: Any) -> list[Entity]:
        found: dict[tuple[str, str], Entity] = {}
        for item in doc.ents:
            text = item.text.strip()
            if not text:
                continue
            key = (normalize(text), item.label_)
            found[key] = Entity(
                text=text,
                label=item.label_,
                start_char=item.start_char,
                end_char=item.end_char,
                source="spacy",
            )

        source_text = doc.text
        for _concept, pattern in _DOMAIN_PATTERNS:
            match = pattern.search(source_text)
            if match:
                key = (normalize(match.group(0)), "CONCEPT_NEUROPSICOLÓGICO")
                found.setdefault(
                    key,
                    Entity(
                        text=match.group(0),
                        label="CONCEPT_NEUROPSICOLÓGICO",
                        start_char=match.start(),
                        end_char=match.end(),
                        source="domain_lexicon",
                    ),
                )
        return sorted(
            found.values(),
            key=lambda item: (
                item.start_char if item.start_char is not None else 10**9,
                item.label,
                item.text.casefold(),
            ),
        )[:1000]

    def _extract_keyphrases(self, doc: Any) -> list[Keyphrase]:
        candidates: Counter[str] = Counter()
        display: dict[str, str] = {}
        stopwords = NORMALIZED_STOPWORDS

        try:
            chunks = doc.noun_chunks
            for chunk in chunks:
                phrase = re.sub(r"\s+", " ", chunk.text).strip(" ,;:.()[]{}")
                words = content_words(phrase)
                if not words or len(phrase) > 160:
                    continue
                key = normalize(phrase)
                candidates[key] += 1
                display.setdefault(key, phrase)
        except (ValueError, NotImplementedError):
            pass

        # Complementa os chunks com sequências curtas de substantivos, nomes e adjetivos.
        for sentence in doc.sents:
            run: list[str] = []
            for token in sentence:
                eligible = (
                    token.pos_ in {"NOUN", "PROPN", "ADJ"}
                    and not token.is_stop
                    and not token.is_punct
                    and len(token.text) > 2
                )
                if eligible:
                    run.append(token.text)
                    for width in (1, 2, 3):
                        if len(run) >= width:
                            phrase = " ".join(run[-width:])
                            key = normalize(phrase)
                            if key not in stopwords:
                                candidates[key] += 1
                                display.setdefault(key, phrase)
                else:
                    run = []

        if not candidates:
            fallback = Counter(content_words(doc.text))
            for term, count in fallback.items():
                candidates[term] = count
                display[term] = term

        maximum = max(candidates.values(), default=1)
        ranked = sorted(
            candidates.items(),
            key=lambda pair: (
                pair[1] * (1.0 + 0.08 * max(0, len(pair[0].split()) - 1)),
                len(pair[0]),
                pair[0],
            ),
            reverse=True,
        )
        selected: list[Keyphrase] = []
        seen: set[str] = set()
        for key, count in ranked:
            if key in seen or not key.strip():
                continue
            phrase = display.get(key, key)
            if not content_words(phrase) or len(phrase) > 160:
                continue
            selected.append(
                Keyphrase(
                    phrase=phrase,
                    score=min(1.0, count / maximum),
                    frequency=count,
                )
            )
            seen.add(key)
            if len(selected) >= 40:
                break
        return selected

    def _extract_relations(self, doc: Any) -> list[Relation]:
        relations: list[Relation] = []
        seen: set[tuple[str, str, str]] = set()
        for sentence in doc.sents:
            root = sentence.root
            subjects = [child for child in root.children if child.dep_ in _SUBJECT_DEPS]
            objects = [child for child in root.children if child.dep_ in _OBJECT_DEPS]
            if not subjects:
                continue
            for subject in subjects:
                for object_token in objects:
                    subject_text = self._compact_span(subject)
                    object_text = self._compact_span(object_token)
                    predicate = (root.lemma_ or root.text).strip()
                    if not subject_text or not object_text or not predicate:
                        continue
                    key = (
                        normalize(subject_text),
                        normalize(predicate),
                        normalize(object_text),
                    )
                    if key in seen:
                        continue
                    evidence = sentence.text.strip()
                    relations.append(
                        Relation(
                            subject=subject_text,
                            predicate=predicate,
                            object=object_text,
                            evidence=evidence[:1200],
                            confidence=0.55,
                        )
                    )
                    seen.add(key)
                    if len(relations) >= 500:
                        return relations
        return relations

    @staticmethod
    def _compact_span(token: Any) -> str:
        value = re.sub(r"\s+", " ", token.text).strip()
        return value[:300]

    def _rank_paragraphs(
        self,
        text: str,
        sentences: list[str],
        scores: list[float],
    ) -> list[ParagraphMemory]:
        paragraphs = split_paragraphs(text)
        if not paragraphs:
            paragraphs = sentences[: self.settings.max_paragraphs]
        ranked = sorted(
            enumerate(paragraphs),
            key=lambda item: (paragraph_relevance(item[1], sentences, scores), -item[0]),
            reverse=True,
        )
        return [
            ParagraphMemory(rank=rank, text=paragraph[:5000])
            for rank, (_, paragraph) in enumerate(ranked[: self.settings.max_paragraphs], start=1)
        ]

    @staticmethod
    def _build_concept_graph(
        entities: list[Entity],
        keyphrases: list[Keyphrase],
        relations: list[Relation],
    ) -> ConceptGraph:
        nodes: dict[str, ConceptNode] = {}
        labels_to_ids: dict[str, str] = {}

        def add_node(label: str, kind: str) -> str:
            key = normalize(label)
            if not key:
                return ""
            if key in labels_to_ids:
                node_id = labels_to_ids[key]
                prior = nodes[node_id]
                nodes[node_id] = prior.model_copy(update={"occurrences": prior.occurrences + 1})
                return node_id
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
            node_id = f"node-{digest}"
            nodes[node_id] = ConceptNode(
                id=node_id,
                label=label.strip(),
                kind=kind,
                occurrences=1,
            )
            labels_to_ids[key] = node_id
            return node_id

        for entity in entities:
            kind = "concept" if entity.label == "CONCEPT_NEUROPSICOLÓGICO" else "entity"
            add_node(entity.text, kind)
        for keyphrase in keyphrases:
            add_node(keyphrase.phrase, "keyphrase")

        edges: list[ConceptEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for relation in relations:
            source_id = add_node(relation.subject, "entity")
            target_id = add_node(relation.object, "entity")
            if not source_id or not target_id:
                continue
            edge_key = (source_id, target_id, normalize(relation.predicate))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append(
                ConceptEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation.predicate,
                    evidence=relation.evidence,
                    confidence=relation.confidence,
                )
            )
        return ConceptGraph(nodes=list(nodes.values()), edges=edges)

    def _resolve_device(self) -> str:
        if self._device is not None:
            return self._device
        requested = self.settings.device.casefold()
        if requested != "auto":
            self._device = requested
            return requested
        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self._device = "cpu"
        return self._device

    def _summarize(self, text: str) -> str:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise NLPDependencyError(
                "Transformers/PyTorch não instalados; use pip install -e '.[full]'."
            ) from error

        if self._summarizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.settings.summarization_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.settings.summarization_model)
            model = model.to(torch.device(self._resolve_device()))
            model.eval()
            self._summarizer = (tokenizer, model)

        tokenizer, model = self._summarizer
        inputs = tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(model.parameters()).device
        inputs = {name: value.to(model_device) for name, value in inputs.items()}
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=96,
                min_new_tokens=12,
                num_beams=4,
                no_repeat_ngram_size=3,
                do_sample=False,
            )
        summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        if not summary:
            raise NLPError("O modelo não retornou um resumo abstrativo.")
        return summary

    def _embed(self, paragraphs: list[str]) -> list[list[float]]:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise NLPDependencyError(
                "sentence-transformers/PyTorch não instalados; use pip install -e '.[full]'."
            ) from error
        if self._embedder is None:
            self._embedder = SentenceTransformer(
                self.settings.embedding_model,
                device=self._resolve_device(),
            )
            self._embedding_dimension = int(self._embedder.get_sentence_embedding_dimension())
        vectors = self._embedder.encode(
            paragraphs,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(paragraphs):
            raise NLPError("O modelo retornou embeddings em formato inesperado.")
        return array.tolist()
