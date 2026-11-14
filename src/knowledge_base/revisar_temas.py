"""Gera a planilha de revisão manual dos temas.

Por que existe: o agrupamento automático dos 101 temas falhou duas vezes, e
sempre do mesmo jeito — ele reencontra os eixos que o projeto **já tem de
graça**: polaridade (está em `review_score`) e categoria de produto (está em
`categoria`, vinda dos metadados do pedido). O eixo que a camada precisa,
**tipo de problema**, não sai de agrupamento por proximidade vetorial.

Medido em 14/11/2026, agrupando os 101 centroides em 12 por ligação
hierárquica média: 25.000 dos 28.827 documentos caíram em 3 grupos, sendo o
maior "tudo que é elogio" (12.618 docs, nota 4,73) e o segundo "tudo que é
problema de entrega" (8.050 docs, nota 2,14). Os grupos pequenos viraram
`relógio`, `mochila`, `cortina`, `cadeira`, `perfume` — categoria de produto.

Daí a planilha: os 101 temas individuais são bons e validados contra os
metadados, então o que falta é só a junção, e ela é decisão de negócio. A
coluna `macro_proposto` vem em branco de propósito.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"

#: Reviews de exemplo por tema. Três é o suficiente para reconhecer o assunto
#: sem transformar a planilha em leitura de corpus.
N_EXEMPLOS = 3

#: Sugestões de macro-tema, para a revisão ter um vocabulário de partida.
#: Não é taxonomia fechada — a revisão pode criar, fundir ou renomear.
VOCABULARIO_SUGERIDO = [
    "entrega_atrasada", "nao_recebido", "quantidade_errada",
    "produto_errado", "produto_danificado", "qualidade_abaixo",
    "divergente_do_anuncio", "cancelamento", "atendimento",
    "frete_e_retirada", "elogio_entrega", "elogio_produto",
]


def montar() -> pd.DataFrame:
    corpus = pd.read_parquet(PROCESSED / "corpus.parquet")
    mapa = pd.read_parquet(PROCESSED / "corpus_temas.parquet")
    rotulos = pd.read_csv(PROCESSED / "temas_rotulos.csv")

    df = corpus.merge(mapa, on="review_id", how="inner")
    palavras = dict(zip(rotulos.tema, rotulos.palavras.fillna("")))

    base_nota = df.review_score.mean()
    base_atraso = df.entrega_atrasada.mean() * 100

    linhas = []
    for tema, sub in df[df.tema != -1].groupby("tema"):
        # Exemplos: os mais longos do tema. Texto longo descreve o problema;
        # texto curto ("não gostei") não deixa reconhecer o assunto.
        exemplos = (
            sub.nlargest(N_EXEMPLOS, "n_palavras").texto
            .str.replace(r"\s+", " ", regex=True).str.slice(0, 160).tolist()
        )
        atraso = sub.entrega_atrasada.mean() * 100
        linhas.append({
            "tema": tema,
            "n_docs": len(sub),
            "nota_media": round(sub.review_score.mean(), 2),
            # Os dois deltas são o que permite classificar sem ler: tema com
            # atraso muito acima da base é problema de entrega; com atraso na
            # base e nota baixa é problema de produto.
            "delta_nota": round(sub.review_score.mean() - base_nota, 2),
            "atraso_pct": round(atraso, 1),
            "delta_atraso_pp": round(atraso - base_atraso, 1),
            "categoria_dominante": (
                sub.categoria.mode().iat[0] if sub.categoria.notna().any() else ""
            ),
            "pct_categoria_dominante": round(
                sub.categoria.value_counts(normalize=True).iat[0] * 100, 1
            ) if sub.categoria.notna().any() else 0.0,
            "palavras": palavras.get(tema, ""),
            "macro_proposto": "",
            **{f"exemplo_{i + 1}": (exemplos[i] if i < len(exemplos) else "")
               for i in range(N_EXEMPLOS)},
        })

    revisao = pd.DataFrame(linhas).sort_values("n_docs", ascending=False)
    print(f"temas na planilha                {len(revisao)}")
    print(f"documentos cobertos              {revisao.n_docs.sum():,}")
    print(f"referência: nota {base_nota:.2f}, atraso {base_atraso:.1f}%")
    return revisao


def main() -> pd.DataFrame:
    revisao = montar()
    destino = PROCESSED / "temas_para_revisar.csv"
    revisao.to_csv(destino, index=False, encoding="utf-8-sig")  # -sig: Excel
    print(f"\ngravado em {destino.relative_to(RAIZ)}")
    print(f"preencher a coluna `macro_proposto`. vocabulário sugerido:")
    for v in VOCABULARIO_SUGERIDO:
        print(f"  {v}")

    print("\nos 12 maiores, para conferência:")
    cols = ["tema", "n_docs", "nota_media", "atraso_pct", "delta_atraso_pp", "palavras"]
    for _, r in revisao.head(12).iterrows():
        print(f"  {r.tema:>3} | {r.n_docs:>5,} docs | nota {r.nota_media:4.2f} | "
              f"atraso {r.atraso_pct:5.1f}% ({r.delta_atraso_pp:+5.1f} p.p.) | "
              f"{r.palavras[:62]}")
    return revisao


if __name__ == "__main__":
    main()
