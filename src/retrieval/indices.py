"""Construção dos dois índices da busca híbrida.

Requisito 2 do Tech Challenge. São dois índices sobre o MESMO corpus, cada um
lendo uma coluna diferente — e é por isso que a limpeza tem rotas separadas:

    texto      -> embedding -> FAISS   (acha paráfrase)
    texto_bm25 -> tokens    -> BM25    (acha palavra literal)

Por que dois: a perna vetorial acha "chegou atrasado" quando a pergunta diz
"demorou", sem nenhuma palavra em comum. A perna lexical acha "Correios",
"nota fiscal" e nome de transportadora, que o embedding dilui em sentido
genérico de logística. Cada uma cobre o buraco da outra.

FAISS local, sem serviço gerenciado (decisão de 12/11): 39.388 vetores de 384
dimensões são 60 MB, cabem em memória, e qualquer avaliador reproduz sem
credencial. Azure AI Search e pgvector ficam citados como caminho de produção.

`IndexFlatIP` é busca exata por produto interno, não aproximada. Com vetores
normalizados, produto interno **é** similaridade de cosseno. Não usa HNSW de
propósito: em 39 mil vetores a busca exata leva milissegundos, e índice
aproximado introduziria recall imperfeito sem ganho perceptível. (No Maestro,
onde o volume é outro, HNSW se justifica.)
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"
INDEX = RAIZ / "data" / "index"

MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

#: Parâmetros do BM25. Os defaults da literatura (Robertson & Walker): k1
#: controla a saturação da repetição do termo, b o peso da normalização por
#: comprimento. b=0,75 é relevante aqui: com o piso de 2 palavras no índice,
#: documento curto pontua alto, e é esse parâmetro que governa o efeito.
BM25_K1 = 1.5
BM25_B = 0.75


def _log(etapa: str, valor: object) -> None:
    print(f"  {etapa:<46} {valor}")


def carregar_corpus() -> pd.DataFrame:
    print("\n[1] Corpus")
    df = pd.read_parquet(PROCESSED / "corpus.parquet").reset_index(drop=True)
    _log("documentos", f"{len(df):,}")
    _log("sem token na coluna lexical",
         int((df.texto_bm25.fillna("").str.strip() == "").sum()))
    return df


def construir_faiss(df: pd.DataFrame) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    print("\n[2] Índice vetorial (FAISS)")
    cache = PROCESSED / "emb_corpus_completo.npy"
    if cache.exists() and len(np.load(cache, mmap_mode="r")) == len(df):
        emb = np.load(cache)
        _log("embeddings do cache", emb.shape)
    else:
        modelo = SentenceTransformer(MODELO_EMBEDDING)
        emb = modelo.encode(df.texto.tolist(), show_progress_bar=False,
                            batch_size=64).astype("float32")
        np.save(cache, emb)
        _log("embeddings calculados", emb.shape)

    emb = np.ascontiguousarray(emb, dtype="float32")
    # Normalizar é o que transforma produto interno em cosseno. Sem isso, um
    # documento de vetor longo venceria por magnitude, não por sentido.
    faiss.normalize_L2(emb)

    indice = faiss.IndexFlatIP(emb.shape[1])
    indice.add(emb)
    INDEX.mkdir(parents=True, exist_ok=True)
    faiss.write_index(indice, str(INDEX / "faiss_corpus.index"))
    _log("vetores no índice", f"{indice.ntotal:,}")
    _log("gravado", "data/index/faiss_corpus.index")


def construir_bm25(df: pd.DataFrame) -> None:
    from rank_bm25 import BM25Okapi

    print("\n[3] Índice lexical (BM25)")
    # A coluna já vem stemizada e com a negação intacta, da rota lexical da
    # limpeza. Aqui só reparte: o custo de stemizar 39 mil textos foi pago uma
    # vez na ingestão, de propósito, para não entrar na latência da pergunta.
    corpus_tokens = [t.split() for t in df.texto_bm25.fillna("")]
    bm25 = BM25Okapi(corpus_tokens, k1=BM25_K1, b=BM25_B)

    INDEX.mkdir(parents=True, exist_ok=True)
    with open(INDEX / "bm25_corpus.pkl", "wb") as f:
        pickle.dump(bm25, f)

    tamanhos = np.array([len(t) for t in corpus_tokens])
    _log("documentos", f"{len(corpus_tokens):,}")
    _log("tokens por documento (mediana / p90)",
         f"{np.median(tamanhos):.0f} / {np.quantile(tamanhos, .9):.0f}")
    _log("vocabulário", f"{len(bm25.idf):,} termos")
    _log("gravado", "data/index/bm25_corpus.pkl")


def salvar_mapa(df: pd.DataFrame) -> None:
    """Posição no índice -> review_id e metadados de citação.

    FAISS devolve posição inteira, não identificador. Sem este mapa não há
    como citar `review_id`, que é o requisito de rastreabilidade.
    """
    print("\n[4] Mapa de citação")
    cols = ["review_id", "texto", "review_score", "data_review", "categoria",
            "entrega_atrasada", "atraso_dias", "customer_state", "n_palavras"]
    mapa = df[cols].copy()
    mapa["posicao"] = np.arange(len(mapa))
    mapa.to_parquet(INDEX / "mapa_citacao.parquet", index=False)
    _log("linhas", f"{len(mapa):,}")
    _log("gravado", "data/index/mapa_citacao.parquet")


def main() -> None:
    df = carregar_corpus()
    construir_faiss(df)
    construir_bm25(df)
    salvar_mapa(df)
    print("\nÍndices prontos.")


if __name__ == "__main__":
    main()
