"""Normalização de texto — duas rotas com destinos diferentes.

A aula 02 entrega um pipeline único (`limpar_normalizar` + stop words + RSLP).
Aplicá-lo ao texto que vira embedding inverteria o sentido do corpus: 26% das
reviews do Olist contêm marcador de negação, e `não`/`nem`/`sem` estão entre as
207 stop words do NLTK em português. A saída do próprio slide 57 mostra o efeito:

    'Não gostei, veio quebrado e atrasado.' -> ['gost', 'vei', 'quebr', 'atras']

Daí a separação:

    normalizar_para_indexacao()  -> texto que será vetorizado (retrieval)
    pipeline_nlp_analitico()     -> tokens para contagem e análise do corpus
"""

from __future__ import annotations

import re
import unicodedata

from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

# Contrações do slide 15, restritas às que ocorrem no corpus Olist.
# Medido em 40.950 reviews: 1.391 (3,40%) contêm pelo menos uma.
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
