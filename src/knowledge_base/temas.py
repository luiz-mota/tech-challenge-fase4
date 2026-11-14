"""Camada de agregação por tema — responde perguntas de frequência.

Por que esta camada existe: a busca híbrida devolve as reviews mais
*parecidas* com a pergunta, não as mais *frequentes*. Para responder "quais os
principais problemas?" é preciso agrupar as reviews por assunto e contar. Isso
é modelagem de tópico, não recuperação.

O que foi medido para chegar nestes parâmetros (13/11/2026):

  1. min_topic_size=150 + UMAP default -> 4 temas, 0,6% de outliers, e os dois
     maiores eram "nota 2,28" e "nota 4,73". O modelo separou POLARIDADE, não
     assunto — e polaridade já é grátis, está em `review_score`. Causa: o eixo
     de maior variância num corpus de review é gostei × não gostei, e o UMAP
     com vizinhança global achata todo o resto nele.

  2. min_topic_size=50 + n_neighbors=10 -> 62 temas, 42,9% de outliers, e aí
     os temas discriminaram assunto. A prova é independente do texto: os temas
     de entrega ficaram com 31–56% de entrega atrasada contra 10,4% do corpus,
     enquanto os temas de produto ("cor errada", "caixa amassada", "foto
     diferente") ficaram em 3–4% com a MESMA nota ruim. A nota não distingue
     esses grupos; o tema distingue.

  3. 42,9% de outliers era alto demais para um entregável. Tratado por duas
     vias, nesta ordem: mais granularidade (min_topic_size 50 -> 30) e depois
     realocação dos outliers restantes só quando há proximidade suficiente —
     ver REALOCACAO_MIN_SIMILARIDADE.

Nomes de vendedor: a Olist anonimizou lojas com nomes de Game of Thrones, e
`lannister` aparece em 3,08% do corpus. Como nome próprio frequente, ele vence
a disputa de c-TF-IDF e vira rótulo de tema — "lannister, loja, contato" não é
assunto, é vendedor. Entram como stop word de domínio, apenas na NOMEAÇÃO.
As reviews continuam no índice e continuam citáveis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.knowledge_base.limpeza import STOP_WORDS_PT  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"

MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

#: Piso de palavras da modelagem, maior que o do índice (2). Decisão de
#: 12/11: tópico sobre "muito bom" produz lixo. Cobre 73,2% do corpus.
PISO_PALAVRAS = 6

#: Tamanho mínimo de um tema. Abaixo disso não é "problema principal" de nada.
MIN_TOPIC_SIZE = 30

#: Vizinhança do UMAP. O default (15) é global e achata tudo no eixo de
#: polaridade; 10 preserva estrutura local, que é onde o assunto vive.
UMAP_N_NEIGHBORS = 10

#: Similaridade de cosseno mínima para realocar um outlier no tema mais
#: próximo. Abaixo dela a review fica sem tema, de propósito: "não se encaixa
#: em padrão recorrente" é uma resposta honesta, melhor que encaixe forçado.
REALOCACAO_MIN_SIMILARIDADE = 0.55

#: Quantidade de macro-temas. Os ~100 temas do BERTopic são granulares demais
#: para uma resposta: "chegou antes do prazo" se parte em 4 temas distintos, e
#: nenhum pareceria "principal". Agrupados por proximidade dos centroides.
N_MACRO_TEMAS = 12

#: Nomes de vendedor anonimizados pela Olist (casas de Game of Thrones).
#: Removidos apenas da nomeação dos temas, nunca do índice.
NOMES_VENDEDOR = frozenset({"lannister", "stark", "targaryen", "baratheon"})

#: Palavras que descrevem o ato de avaliar, não o assunto avaliado. Sem elas
#: fora, vários temas se chamariam "recomendo, produto, comprei".
RUIDO_DOMINIO = frozenset({
    "produto", "produtos", "comprei", "compra", "comprar", "recomendo",
    "loja", "lojas", "site", "pedido", "recebi", "veio", "chegou",
})

STOP_WORDS_TEMAS = list(STOP_WORDS_PT | NOMES_VENDEDOR | RUIDO_DOMINIO)


def _log(etapa: str, valor: object) -> None:
    print(f"  {etapa:<48} {valor}")


def carregar_corpus_modelagem() -> pd.DataFrame:
    print("\n[1] Corpus da modelagem")
    df = pd.read_parquet(PROCESSED / "corpus.parquet")
    mod = df[df.n_palavras >= PISO_PALAVRAS].copy()
    _log("corpus completo", f"{len(df):,}")
    _log(f"com >= {PISO_PALAVRAS} palavras",
         f"{len(mod):,} ({len(mod) / len(df) * 100:.1f}%)")
    return mod


def obter_embeddings(mod: pd.DataFrame):
    """Embeddings com cache em disco: 6 min de CPU não devem ser repagos."""
    from sentence_transformers import SentenceTransformer

    print("\n[2] Embeddings")
    cache = PROCESSED / f"emb_modelagem_piso{PISO_PALAVRAS}.npy"
    modelo = SentenceTransformer(MODELO_EMBEDDING)
    if cache.exists():
        emb = np.load(cache)
        if len(emb) == len(mod):
            _log("lidos do cache", f"{emb.shape}")
            return modelo, emb
        _log("cache com tamanho divergente, recalculando",
             f"{len(emb)} != {len(mod)}")
    emb = modelo.encode(mod.texto.tolist(), show_progress_bar=False, batch_size=64)
    np.save(cache, emb)
    _log("calculados e gravados", f"{emb.shape}")
    return modelo, emb


def treinar(mod: pd.DataFrame, emb) -> tuple:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    print("\n[3] Modelagem de temas")
    vectorizer = CountVectorizer(
        stop_words=STOP_WORDS_TEMAS, min_df=5, ngram_range=(1, 2)
    )
    umap_local = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS, n_components=5, min_dist=0.0,
        metric="cosine", random_state=42,
    )
    modelo_temas = BERTopic(
        language="multilingual",
        vectorizer_model=vectorizer,
        umap_model=umap_local,
        min_topic_size=MIN_TOPIC_SIZE,
        calculate_probabilities=False,
        verbose=False,
    )
    temas, _ = modelo_temas.fit_transform(mod.texto.tolist(), embeddings=emb)
    temas = np.asarray(temas)
    _log("temas encontrados", len(set(temas)) - (1 if -1 in temas else 0))
    _log("outliers antes da realocação",
         f"{int((temas == -1).sum()):,} ({(temas == -1).mean() * 100:.1f}%)")
    return modelo_temas, temas


def realocar_outliers(temas: np.ndarray, emb) -> np.ndarray:
    """Outlier vai para o tema mais próximo, mas só se estiver perto o bastante.

    Alternativa descartada: `BERTopic.reduce_outliers`, que realoca TODOS. Dá
    cobertura de 100% ao preço de afirmar tema onde não há — e a contagem por
    tema é justamente o número que a resposta vai citar.
    """
    print("\n[4] Realocação de outliers")
    temas = temas.copy()
    fora = temas == -1
    if not fora.any():
        _log("nada a realocar", 0)
        return temas

    # Centroide de cada tema, em vetores normalizados: o produto interno
    # passa a ser a similaridade de cosseno.
    norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    rotulos = sorted(t for t in set(temas.tolist()) if t != -1)
    centroides = np.vstack([norm[temas == t].mean(axis=0) for t in rotulos])
    centroides /= np.linalg.norm(centroides, axis=1, keepdims=True)

    sim = norm[fora] @ centroides.T
    melhor = sim.argmax(axis=1)
    escore = sim.max(axis=1)
    aceito = escore >= REALOCACAO_MIN_SIMILARIDADE

    idx = np.flatnonzero(fora)
    temas[idx[aceito]] = np.array(rotulos)[melhor[aceito]]

    _log(f"realocados (similaridade >= {REALOCACAO_MIN_SIMILARIDADE})",
         f"{int(aceito.sum()):,} de {int(fora.sum()):,}")
    _log("similaridade mediana dos realocados",
         f"{np.median(escore[aceito]):.3f}" if aceito.any() else "-")
    _log("mantidos sem tema",
         f"{int((~aceito).sum()):,} ({(temas == -1).mean() * 100:.1f}% do corpus)")
    return temas


def agrupar_macro_temas(mod: pd.DataFrame, emb) -> pd.DataFrame:
    """Junta os temas granulares em macro-temas por proximidade de centroide.

    Por que é necessário: os temas 6, 12, 14 e 24 são todos "chegou antes do
    prazo" — 2.029 reviews partidas em 4 fatias. A pergunta "quais os
    principais problemas?" exige contagem sobre assunto de negócio, e nenhuma
    fatia isolada pareceria principal.

    Ligação hierárquica média sobre os centroides, distância de cosseno. A
    proposta é automática e **deve ser revisada à mão** — o algoritmo agrupa
    por proximidade vetorial, que não é a mesma coisa que categoria de
    negócio. Os rótulos saem de c-TF-IDF sobre o macro-tema inteiro, não da
    soma dos rótulos das partes.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("\n[5] Macro-temas")
    rotulos = sorted(t for t in mod.tema.unique().tolist() if t != -1)
    norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    pos = {t: i for i, t in enumerate(rotulos)}
    centroides = np.vstack([norm[(mod.tema == t).to_numpy()].mean(axis=0)
                            for t in rotulos])
    centroides /= np.linalg.norm(centroides, axis=1, keepdims=True)

    ligacao = linkage(pdist(centroides, metric="cosine"), method="average")
    grupos = fcluster(ligacao, t=N_MACRO_TEMAS, criterion="maxclust")
    _log(f"{len(rotulos)} temas agrupados em", f"{len(set(grupos))} macro-temas")

    mod["macro"] = mod.tema.map(
        lambda t: -1 if t == -1 else int(grupos[pos[t]])
    )

    # c-TF-IDF do macro-tema: cada grupo vira um documento gigante, e o termo
    # que o distingue dos outros grupos é o que o nomeia.
    macros = sorted(m for m in mod.macro.unique().tolist() if m != -1)
    corpora = [" ".join(mod.loc[mod.macro == m, "texto"].tolist()) for m in macros]
    tfidf = TfidfVectorizer(stop_words=STOP_WORDS_TEMAS, ngram_range=(1, 2),
                            max_features=20000, sublinear_tf=True)
    X = tfidf.fit_transform(corpora)
    vocab = np.array(tfidf.get_feature_names_out())

    linhas = []
    for i, m in enumerate(macros):
        sub = mod[mod.macro == m]
        pesos = X[i].toarray().ravel()
        palavras = ", ".join(vocab[pesos.argsort()[::-1][:8]])
        linhas.append({
            "macro": m,
            "n_docs": len(sub),
            "n_temas": sub.tema.nunique(),
            "nota_media": round(sub.review_score.mean(), 2),
            "atraso_pct": round(sub.entrega_atrasada.mean() * 100, 1),
            "palavras": palavras,
            "temas": ", ".join(str(t) for t in sorted(sub.tema.unique()) if t != -1),
        })
    return pd.DataFrame(linhas).sort_values("n_docs", ascending=False)


def relatorio_macro(proposta: pd.DataFrame, mod: pd.DataFrame) -> None:
    print(f"\n  referência do corpus: nota {mod.review_score.mean():.2f}, "
          f"atraso {mod.entrega_atrasada.mean() * 100:.1f}%\n")
    print(f"  {'macro':<7}{'docs':>7}{'temas':>7}{'nota':>6}{'atraso%':>9}  palavras")
    print("  " + "-" * 100)
    for _, r in proposta.iterrows():
        print(f"  {r.macro:<7}{r.n_docs:>7,}{r.n_temas:>7}{r.nota_media:>6.2f}"
              f"{r.atraso_pct:>8.1f}%  {r.palavras}")
    sem = int((mod.macro == -1).sum())
    print(f"  {'-1':<7}{sem:>7,}{'-':>7}"
          f"{mod[mod.macro == -1].review_score.mean():>6.2f}"
          f"{mod[mod.macro == -1].entrega_atrasada.mean() * 100:>8.1f}%  (sem tema)")


def relatorio(mod: pd.DataFrame, modelo_temas, n: int = 20) -> None:
    print("\n[5] Temas")
    print(f"  referência do corpus: nota {mod.review_score.mean():.2f}, "
          f"atraso {mod.entrega_atrasada.mean() * 100:.1f}%\n")
    print(f"  {'tema':<5}{'docs':>7}{'nota':>6}{'atraso%':>9}  palavras")
    print("  " + "-" * 96)
    contagem = mod.tema.value_counts()
    ordem = [-1] + [t for t in contagem.index if t != -1][:n]
    for tema in ordem:
        sub = mod[mod.tema == tema]
        if sub.empty:
            continue
        palavras = ("(sem tema)" if tema == -1 else
                    ", ".join(w for w, _ in modelo_temas.get_topic(tema)[:7]))
        print(f"  {tema:<5}{len(sub):>7,}{sub.review_score.mean():>6.2f}"
              f"{sub.entrega_atrasada.mean() * 100:>8.1f}%  {palavras}")

    com_tema = mod[mod.tema != -1]
    print(f"\n  dispersão da nota entre temas      "
          f"{com_tema.groupby('tema').review_score.mean().std():.2f}")
    print(f"  dispersão do atraso entre temas    "
          f"{com_tema.groupby('tema').entrega_atrasada.mean().std() * 100:.1f} p.p.")


def salvar(mod: pd.DataFrame, modelo_temas, proposta: pd.DataFrame) -> None:
    print("\n[7] Gravação")
    proposta.to_csv(PROCESSED / "macro_temas.csv", index=False, encoding="utf-8")
    _log("proposta de macro-temas (REVISAR À MÃO)", "data/processed/macro_temas.csv")

    destino = PROCESSED / "corpus_temas.parquet"
    mod[["review_id", "tema", "macro"]].to_parquet(destino, index=False)
    _log("mapa review_id -> tema", destino.relative_to(RAIZ))

    rotulos = (
        modelo_temas.get_topic_info()[["Topic", "Count", "Name"]]
        .rename(columns={"Topic": "tema", "Count": "n_original", "Name": "rotulo"})
    )
    rotulos["palavras"] = [
        ", ".join(w for w, _ in modelo_temas.get_topic(t)[:10]) if t != -1 else ""
        for t in rotulos.tema
    ]
    rotulos["n_final"] = rotulos.tema.map(mod.tema.value_counts()).fillna(0).astype(int)
    rotulos.to_csv(PROCESSED / "temas_rotulos.csv", index=False, encoding="utf-8")
    _log("rótulos", "data/processed/temas_rotulos.csv")

    modelo_temas.save(
        str(PROCESSED / "modelo_temas"),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=MODELO_EMBEDDING,
    )
    _log("modelo BERTopic", "data/processed/modelo_temas/")


def main() -> pd.DataFrame:
    mod = carregar_corpus_modelagem()
    _, emb = obter_embeddings(mod)
    modelo_temas, temas = treinar(mod, emb)
    mod["tema"] = realocar_outliers(temas, emb)
    proposta = agrupar_macro_temas(mod, emb)
    relatorio_macro(proposta, mod)
    relatorio(mod, modelo_temas)
    salvar(mod, modelo_temas, proposta)
    return mod


if __name__ == "__main__":
    main()
