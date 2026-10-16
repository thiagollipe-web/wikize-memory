from wikize_memory.nlp.features import (
    extractive_summary,
    normalize,
    split_paragraphs,
    split_sentences,
)


def test_normalize_removes_accents_for_deduplication():
    assert normalize("Memória de Trabalho") == "memoria de trabalho"


def test_split_sentences_preserves_sentence_text():
    sentences = split_sentences(
        "A memória apoia a aprendizagem. O hipocampo participa da consolidação."
    )
    assert len(sentences) == 2
    assert sentences[0].endswith("aprendizagem.")


def test_extractive_summary_returns_salient_sentences_in_source_order():
    sentences = [
        "O artigo apresenta uma breve introdução.",
        "A memória participa da aprendizagem e da cognição.",
        "A memória permite adquirir, armazenar e recuperar informações.",
    ]
    summary, claims = extractive_summary(sentences, limit=2)
    assert summary
    assert len(claims) == 2
    assert sentences[1] in summary
    assert sentences[2] in summary


def test_split_paragraphs_discards_section_headings():
    text = (
        "== Introdução ==\n"
        "A memória é um processo cognitivo que permite guardar informações e experiências.\n\n"
        "== Sistemas ==\n"
        "O hipocampo participa da consolidação de memórias de longo prazo."
    )
    paragraphs = split_paragraphs(text)
    assert len(paragraphs) == 2
    assert all(not paragraph.startswith("==") for paragraph in paragraphs)
