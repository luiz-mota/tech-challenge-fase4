"""Aplica os temas e macro-temas ao corpus INTEIRO.

Treinar e aplicar não precisam ter a mesma cobertura, e aqui de propósito não
têm:

    treino   -> 28.827 docs (piso de 6 palavras)
    aplicação -> 39.388 docs (o corpus inteiro)

Por que o treino usa piso 6: modelagem de tópico sobre "muito bom" produz
lixo — o texto curto não tem o que agrupar. Por que a aplicação não usa:
depois de treinado, o modelo tem centroides, e medir a distância de um
documento curto a um centroide é perfeitamente definido.

O defeito que isso corrige (medido em 14/11): a avaliação de recall marcava
"ERROU" em perguntas que a busca acertava. Causa: perguntas curtas recuperam
documentos curtos, documentos curtos não tinham tema, e o gabarito não
conseguia pontuar. Três das cinco falhas eram isso — defeito da medição, não
da busca.

O piso de similaridade da atribuição é o mesmo da realocação de outliers
(0,55), pela mesma razão: documento que não pertence a tema nenhum fica sem
tema. Rótulo forçado produz contagem errada, e a contagem é o que o sistema
afirma ao usuário.

Saída: data/processed/corpus_temas.parquet (review_id, tema, macro, origem)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.knowledge_base.mapa_macro import PROPOSTA  # noqa: E402
from src.knowledge_base.temas import (  # noqa: E402
    PISO_PALAVRAS,
    REALOCACAO_MIN_SIMILARIDADE,
)

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"

MAPA_MACRO = {t: macro for macro, temas in PROPOSTA.items() for t in temas}

#: Macro-temas que NÃO entram na contagem de "principal problema", cada um por
#: um motivo diferente: `misto` porque a review trata de vários assuntos e
#: nenhum rótulo é verdade; `sem_avaliacao_ainda` porque não há avaliação;
#: `sem_tema` porque o documento não pertence a padrão recorrente.
FORA_DA_CONTAGEM = frozenset({"misto", "sem_avaliacao_ainda", "sem_tema"})

#: Macro-temas que são elogio. Separados para a contagem de problemas não
#: misturar as duas coisas.
ELOGIOS = frozenset({"elogio_entrega", "elogio_produto", "elogio_loja_atendimento"})


def _log(etapa: str, valor: object) -> None:
    print(f"  {etapa:<48} {valor}")


def _carregar_treino() -> pd.DataFrame:
    """Só os documentos que o BERTopic realmente treinou.

    Este módulo grava em `corpus_temas.parquet` e também o lê — e isso tinha
    um defeito de idempotência descoberto em 15/11: na segunda execução, os
    10.211 documentos atribuídos por centroide entravam no cálculo dos
    centroides, que então se deslocavam. Rodar duas vezes dava resultado
    diferente de rodar uma, e o pipeline que o README diz ser reprodutível
    não era.

    A correção não exige retreinar nada: a coluna `origem` recupera
    exatamente o conjunto de treino do próprio arquivo de saída. Enquanto ela
    não existe — primeira execução, saída crua do `temas.py` — o arquivo
    inteiro É o treino.

    Um espelho do treino é gravado em `temas_treino.parquet` para que a
    dependência fique explícita e não dependa de uma coluna de controle.
    """
    espelho = PROCESSED / "temas_treino.parquet"
    d = pd.read_parquet(PROCESSED / "corpus_temas.parquet")
    if "origem" in d.columns:
        d = d[d.origem == "treino"][["review_id", "tema", "macro"]].copy()
    if not espelho.exists():
        d.to_parquet(espelho, index=False)
    return d


def main() -> pd.DataFrame:
    print("\n[1] Entradas")
    corpus = pd.read_parquet(PROCESSED / "corpus.parquet").reset_index(drop=True)
    treino = _carregar_treino()
    emb_todos = np.load(PROCESSED / "emb_corpus_completo.npy")
    _log("corpus", f"{len(corpus):,}")
    _log("já com tema (do treino)", f"{len(treino[treino.tema != -1]):,}")
    _log("embeddings do corpus completo", emb_todos.shape)
    if len(emb_todos) != len(corpus):
        raise SystemExit("embeddings e corpus com tamanhos diferentes")

    print("\n[2] Centroides dos temas")
    norm = emb_todos / np.linalg.norm(emb_todos, axis=1, keepdims=True)
    pos_por_id = {rid: i for i, rid in enumerate(corpus.review_id)}
    tema_por_id = dict(zip(treino.review_id, treino.tema))

    rotulos = sorted(t for t in set(tema_por_id.values()) if t != -1)
    centroides = []
    for t in rotulos:
        idx = [pos_por_id[r] for r, tt in tema_por_id.items()
               if tt == t and r in pos_por_id]
        centroides.append(norm[idx].mean(axis=0))
    centroides = np.vstack(centroides)
    centroides /= np.linalg.norm(centroides, axis=1, keepdims=True)
    _log("temas", len(rotulos))

    print("\n[3] Atribuição ao corpus inteiro")
    sim = norm @ centroides.T
    melhor = sim.argmax(axis=1)
    escore = sim.max(axis=1)
    aceito = escore >= REALOCACAO_MIN_SIMILARIDADE

    temas = np.full(len(corpus), -1)
    temas[aceito] = np.array(rotulos)[melhor[aceito]]

    # Quem estava no treino mantém o tema do treino: o agrupamento viu o
    # documento junto dos seus vizinhos, o centroide só vê a média.
    do_treino = corpus.review_id.map(tema_por_id)
    tinha = do_treino.notna() & (do_treino != -1)
    temas = np.where(tinha, do_treino.fillna(-1).to_numpy(), temas).astype(int)

    origem = np.where(tinha, "treino",
                      np.where(aceito, "centroide", "sem_tema"))
    _log("do treino (mantidos)", f"{int(tinha.sum()):,}")
    novos = int((origem == "centroide").sum())
    _log(f"novos por centroide (sim >= {REALOCACAO_MIN_SIMILARIDADE})", f"{novos:,}")
    _log("similaridade mediana dos novos",
         f"{np.median(escore[origem == 'centroide']):.3f}" if novos else "-")
    _log("sem tema", f"{int((origem == 'sem_tema').sum()):,} "
                     f"({(origem == 'sem_tema').mean() * 100:.1f}%)")

    print("\n[4] Macro-temas")
    saida = pd.DataFrame({
        "review_id": corpus.review_id,
        "tema": temas,
        "macro": [MAPA_MACRO.get(t, "sem_tema") for t in temas],
        "origem": origem,
    })
    saida["conta_como_problema"] = ~(
        saida.macro.isin(FORA_DA_CONTAGEM) | saida.macro.isin(ELOGIOS)
    )

    base = corpus.set_index("review_id")
    resumo = (
        saida.join(base[["review_score", "entrega_atrasada", "n_palavras"]],
                   on="review_id")
        .groupby("macro")
        .agg(docs=("tema", "count"), nota=("review_score", "mean"),
             atraso=("entrega_atrasada", "mean"), palavras=("n_palavras", "median"))
        .sort_values("docs", ascending=False)
    )
    total = len(saida)
    print(f"  {'macro-tema':<26}{'docs':>8}{'%':>7}{'nota':>7}{'atraso':>8}{'palavras':>10}")
    print("  " + "-" * 68)
    for nome, r in resumo.iterrows():
        print(f"  {nome:<26}{int(r.docs):>8,}{r.docs / total * 100:>6.1f}%"
              f"{r.nota:>7.2f}{r.atraso * 100:>7.1f}%{r.palavras:>10.0f}")

    problemas = saida[saida.conta_como_problema]
    print(f"\n  contam como problema: {len(problemas):,} "
          f"({len(problemas) / total * 100:.1f}% do corpus)")

    destino = PROCESSED / "corpus_temas.parquet"
    saida.to_parquet(destino, index=False)
    print(f"  gravado em {destino.relative_to(RAIZ)}")
    return saida


if __name__ == "__main__":
    main()
