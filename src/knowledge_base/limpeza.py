"""Normalização de texto — três rotas com destinos diferentes.

A aula 02 entrega um pipeline único (`limpar_normalizar` + stop words + RSLP).
Aplicá-lo ao texto que vira embedding inverteria o sentido do corpus: 26% das
reviews do Olist contêm marcador de negação, e `não`/`nem`/`sem` estão entre as
207 stop words do NLTK em português. A saída do próprio slide 57 mostra o efeito:

    'Não gostei, veio quebrado e atrasado.' -> ['gost', 'vei', 'quebr', 'atras']

Daí a separação:

    normalizar_para_indexacao()  -> texto que será vetorizado (retrieval denso)
    pipeline_lexical_bm25()      -> tokens para a perna BM25 da busca híbrida
    pipeline_nlp_analitico()     -> tokens para contagem e análise do corpus

As duas primeiras preservam a negação, por motivos diferentes: o embedding
representa sentido, e o BM25 compara palavra com palavra. Só a terceira pode
perdê-la, porque representa frequência — e ali `não` é de fato ruído de alta
frequência.
"""

from __future__ import annotations

import re
import unicodedata

from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

# Contrações do slide 15, restritas às que ocorrem no corpus Olist.
# Medido nos 39.388 documentos do corpus: 1.386 (3,52%) contêm pelo menos uma.
# `pra` responde por 780 delas, `pq` por 198 e `n` por 121.
CONTRACOES = {
    "vc": "você", "tb": "também", "tbm": "também", "pq": "por que",
    "obg": "obrigado", "blz": "beleza", "td": "tudo", "msm": "mesmo",
    "vlw": "valeu", "n": "não", "naum": "não", "mt": "muito", "pra": "para",
}

_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"http\S+|www\.\S+")
_RE_EMAIL = re.compile(r"\S+@\S+")
_RE_PONTUACAO_REPETIDA = re.compile(r"([!?.,])\1+")
_RE_ESPACOS = re.compile(r"\s+")
_RE_TOKEN = re.compile(r"^(\W*)(\w+)(\W*)$", re.UNICODE)


def _expandir_contracoes(texto: str) -> str:
    """Troca gíria por forma plena preservando a pontuação colada ao token."""
    saida = []
    for token in texto.split():
        m = _RE_TOKEN.match(token)
        if m:
            antes, palavra, depois = m.groups()
            expansao = CONTRACOES.get(palavra.lower())
            if expansao is not None:
                token = f"{antes}{expansao}{depois}"
        saida.append(token)
    return " ".join(saida)


def normalizar_para_indexacao(texto: object) -> str:
    """Prepara o texto que será convertido em embedding.

    Remove apenas ruído de superfície. **Preserva** deliberadamente:

    - caixa      -> 3,55% das reviews são inteiramente em maiúsculas (ênfase);
    - acentos    -> 51,24% têm acento; `coração` existe no vocabulário dos
                    tokenizers PT, `coracao` fragmenta em subwords (slide 18);
    - dígitos    -> 10,05% têm dígito, e é neles que mora a evidência
                    ("29 de março", "R$ 25,00", "15 dias de atraso");
    - emoji      -> 0,67%, carregam polaridade;
    - negação    -> 25,9%; sem ela o corpus de reclamações perde o sentido.
    """
    if not isinstance(texto, str):
        return ""
    # NFC: acentos podem chegar decompostos (a + ~) e quebrariam a tokenização.
    texto = unicodedata.normalize("NFC", texto)
    texto = _RE_HTML.sub(" ", texto)
    texto = _RE_URL.sub(" ", texto)
    texto = _RE_EMAIL.sub(" ", texto)
    texto = _expandir_contracoes(texto)
    texto = _RE_PONTUACAO_REPETIDA.sub(r"\1", texto)  # "ÓTIMO!!!!" -> "ÓTIMO!"
    return _RE_ESPACOS.sub(" ", texto).strip()


# --------------------------------------------------------------------------
# Rota analítica — o pipeline da aula 02 (slides 20 e 57), sem alterações de
# mérito. Serve para caracterizar o corpus (frequência de termos, Zipf) e
# alimentará o BoW/TF-IDF da aula 03. NUNCA é usado para gerar embeddings.
# --------------------------------------------------------------------------

STOP_WORDS_PT = set(stopwords.words("portuguese"))
_STEMMER = RSLPStemmer()
_RE_SO_LETRAS = re.compile(r"[^a-záàâãäéèêëíìîïóòôõöúùûüç\s.,!?]")


def limpar_normalizar(texto: object, remove_acentos: bool = False) -> str:
    """`limpar_normalizar` do slide 20, com os dois destinos explicitados."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = _RE_HTML.sub(" ", texto)
    texto = _RE_URL.sub(" ", texto)
    texto = _RE_EMAIL.sub(" ", texto)
    texto = _expandir_contracoes(texto)
    texto = _RE_SO_LETRAS.sub(" ", texto)  # descarta dígitos: aceitável só aqui
    if remove_acentos:
        nfkd = unicodedata.normalize("NFKD", texto)
        texto = "".join(c for c in nfkd if not unicodedata.combining(c))
    return _RE_ESPACOS.sub(" ", texto).strip()


def pipeline_nlp_analitico(textos: list[str]) -> list[list[str]]:
    """`pipeline_nlp` do slide 57: texto PT -> tokens sem stop word, stemizados."""
    resultado = []
    for texto in textos:
        texto = limpar_normalizar(texto)
        tokens = word_tokenize(texto, language="portuguese")
        tokens = [t for t in tokens if t not in STOP_WORDS_PT and len(t) > 2]
        resultado.append([_STEMMER.stem(t) for t in tokens])
    return resultado


# --------------------------------------------------------------------------
# Rota lexical — alimenta a perna BM25 da busca híbrida.
#
# É a rota analítica acima com UMA diferença, que existe porque o BM25 compara
# palavra com palavra: a negação permanece. Sem ela, buscar "não recebi" casaria
# com "recebi" com a mesma força, justamente na perna que existe para ser
# literal. É a rota analítica que pode perder a negação (ela conta frequência,
# e ali `não` é ruído de alta frequência); esta, não.
#
# Três ajustes foram necessários, todos medidos no NLTK 207 stop words pt:
#
#   1. `não`, `nem`, `sem` e `mas` SÃO stop words -> saem do conjunto removido;
#   2. `nunca`, `jamais`, `nada`, `nenhum` NÃO são stop words e sobreviveriam,
#      mas o RSLP os deforma (`nunca`->`nunc`, `jamais`->`jamal`). Deformação
#      consistente não quebra o casamento, mas torna o índice ilegível na
#      depuração, então a negação não é stemizada;
#   3. `não` é stop word e `nao` (sem acento) não é. O cliente escreve das duas
#      formas, e o índice ficaria com uma e sem a outra. As variantes são
#      canonizadas para `não` antes do filtro.
#
# Dígitos não entram aqui: `limpar_normalizar` os descarta. É deliberado — a
# evidência quantitativa ("15 dias de atraso") vive na rota de indexação, que
# preserva dígito, e no BM25 número tende a ser ruído (preço, data, nº pedido).
# --------------------------------------------------------------------------

#: Marcadores de negação preservados no índice lexical (regra: nunca remover negação).
NEGACOES = frozenset({
    "não", "nem", "nunca", "sem", "jamais", "nada", "nenhum", "nenhuma", "mas",
})

#: Variantes sem acento ou abreviadas -> forma canônica, para o índice ter um token só.
_VARIANTES_NEGACAO = {
    "nao": "não", "naum": "não", "nunc": "nunca", "jamas": "jamais",
    "ninguem": "nenhum", "nenhun": "nenhum",
}

STOP_WORDS_BM25 = STOP_WORDS_PT - NEGACOES


def pipeline_lexical_bm25(textos: list[str]) -> list[list[str]]:
    """Texto PT -> tokens para o BM25: stemizados, mas com a negação intacta."""
    resultado = []
    for texto in textos:
        texto = limpar_normalizar(texto)
        tokens = word_tokenize(texto, language="portuguese")
        tokens = [_VARIANTES_NEGACAO.get(t, t) for t in tokens]
        tokens = [
            t for t in tokens
            if t in NEGACOES or (t not in STOP_WORDS_BM25 and len(t) > 2)
        ]
        resultado.append([t if t in NEGACOES else _STEMMER.stem(t) for t in tokens])
    return resultado
