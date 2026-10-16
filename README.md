# WikiZé Memory Harvester

Pipeline em Python para coletar artigos da Wikipédia em português, produzir análise linguística estruturada e exportar memórias revisáveis para o THIGAS Synapse v3.

O código trabalha em etapas desacopladas: ingestão pela MediaWiki Action API, análise NLP, validação Pydantic v2, consolidação de memórias e exportação JSON. A interface de terminal usa Typer e Rich.

## Recursos

- Consulta assíncrona com HTTPX, User-Agent configurável, maxlag, retry com backoff exponencial, jitter e suporte a Retry-After.
- Tratamento de página inexistente, redirecionamento de título, desambiguação, timeout, rate limit e paginação.
- Sumarização extrativa local; sumarização abstrativa opcional com csebuetnlp/mT5_multilingual_XLSum.
- NER com spaCy em português (pt_core_news_lg) e identificação suplementar de conceitos de neurociência e psicologia.
- Extração de frases nominais, n-gramas e relações sujeito–predicado–objeto a partir da dependência sintática.
- Embeddings normalizados de parágrafos com Sentence Transformers e paraphrase-multilingual-mpnet-base-v2.
- Esquemas Pydantic v2 para validar artigos, entidades, relações, vetores, memórias e artefatos finais.
- Exportação completa e pacote thigas-synapse-memories versão 3; fatos novos ficam com approved=false para revisão antes do uso.
- Consolidação com arquivo anterior, removendo duplicatas por assunto e fato sem apagar tags ou aliases existentes.
- GitHub Actions para testes e validação, mais criação de GitHub Release com wheel e sdist quando uma tag v* é enviada.

## Instalação

Requer Python 3.11 ou superior.

1. Crie e ative um ambiente virtual.
2. Instale o projeto completo:

    python -m pip install --upgrade pip
    python -m pip install -e ".[full,dev]"
    python -m spacy download pt_core_news_lg

3. Copie .env.example para .env e configure WIKIMEDIA_USER_AGENT com o nome do bot e um contato público do operador. A API Wikimedia exige um User-Agent significativo; não use um valor genérico.
4. Se necessário, ajuste WIKIZEMEMORY_DEVICE para cpu ou cuda. O valor auto escolhe CUDA quando PyTorch a encontra.

A instalação completa baixa PyTorch e os modelos NLP. A primeira execução pode baixar vários gigabytes de pesos para o cache local. Em computadores com pouca memória, os parâmetros --no-abstractive e --no-embeddings evitam carregar mT5 e Sentence Transformers; spaCy ainda precisa do modelo português.

### Windows PowerShell

    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e ".[full,dev]"
    python -m spacy download pt_core_news_lg
    Copy-Item .env.example .env

Edite .env antes de consultar a API.

## Uso

Coletar o artigo e gerar o JSON de análise completo:

    wikize harvest "Memória" --output data/memoria.json

Gerar também o arquivo que o THIGAS Synapse pode importar:

    wikize harvest "Memória" --output data/memoria.json --synapse-output data/memoria-synapse.json

Mesclar o pacote novo com um arquivo de memórias existente:

    wikize harvest "Memória" --output data/memoria.json --merge-existing data/biblioteca.json

Em uma máquina com pouca memória:

    wikize harvest "Memória" --output data/memoria.json --no-abstractive --no-embeddings

Validar qualquer um dos dois JSONs:

    wikize validate data/memoria.json
    wikize validate data/memoria-synapse.json

O mesmo CLI pode ser executado depois da instalação com python -m wikize_memory. O arquivo raiz bot.py é apenas um lançador compatível para esse CLI; ele não contém chaves de API nem inicia um loop durante importação.

## Formato gerado

O JSON completo inclui:

- identificação do artigo, URL canônica, id e timestamp de revisão;
- atribuição da fonte e licença CC BY-SA 4.0;
- resumo extrativo e, quando o modelo está disponível, resumo abstrativo;
- proposições centrais, tópicos, keyphrases, entidades e relações;
- parágrafos selecionados e vetores semânticos normalizados;
- avisos se algum modelo opcional não puder ser carregado;
- um pacote de memórias compatível com o formato THIGAS Synapse v3.

O JSON separado de Synapse contém o campo memories com os campos subject, fact, tags, aliases, approved, confidence e source. Memórias novas entram pendentes de revisão (approved=false). A extração de relações é heurística, e os resumos abstrativos podem reformular o artigo; confira os resultados antes de aprová-los.

O texto usado para análise não é copiado integralmente para o artefato. Os parágrafos selecionados e resumos têm atribuição, URL, id de revisão e licença associados no JSON final. Consulte a licença e os termos do repositório antes de redistribuir o código ou os resultados.

## User-Agent da Wikimedia

Exemplo de configuração em .env:

    WIKIMEDIA_USER_AGENT=WikiMemoryHarvester/1.0 (+https://seu-contato-publico.example)

Substitua o exemplo por uma URL ou e-mail público real. O cliente envia o valor em todas as requisições e respeita Retry-After, limite de tentativas e atraso exponencial.

## Desenvolvimento e publicação

Executar os testes e verificações localmente:

    python -m pip install -e ".[dev]"
    ruff check .
    ruff format --check .
    pytest -q

A workflow CI roda lint, formatação e testes em Python 3.11 e 3.12 nos pushes e pull requests. A workflow Release constrói wheel e sdist, valida os pacotes e cria uma GitHub Release com os artefatos quando uma tag como v0.1.0 é publicada.

O projeto mantém o arquivo LICENSE existente do repositório. Isso é separado da licença CC BY-SA 4.0 do conteúdo da Wikipédia.

## Referências técnicas

- Etiqueta e política de User-Agent da Wikimedia: https://www.mediawiki.org/wiki/API:Etiquette/pt
- Pipelines spaCy para português: https://spacy.io/models/pt/
- Modelos multilíngues Sentence Transformers: https://sbert.net/docs/sentence_transformer/pretrained_models.html
- Modelo de sumarização mT5/XLSum: https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum
