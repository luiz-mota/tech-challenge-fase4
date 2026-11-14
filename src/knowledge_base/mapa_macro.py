"""Proposta de agrupamento dos 101 temas em macro-temas de negócio.

Esta proposta foi montada **lendo** as palavras, a nota média e o
`delta_atraso_pp` de cada tema — não por proximidade vetorial. O agrupamento
automático foi tentado antes e falhou duas vezes do mesmo jeito: ele
reencontra os eixos que o projeto já tem de graça (polaridade, que está em
`review_score`, e categoria de produto, que está em `categoria`), e não o eixo
que a camada precisa, que é **tipo de problema**.

O critério de leitura, nesta ordem:

  1. `delta_atraso_pp` muito positivo (+25 p.p. ou mais) separa "não recebeu"
     de "recebeu com problema" — é metadado, não opinião;
  2. nota média separa reclamação de elogio;
  3. as palavras dizem qual problema é.

A coluna `confianca` marca onde a leitura foi ambígua. São 13 temas: comece
por eles na revisão, porque o resto é inequívoco.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"

#: Macro-tema -> temas que o compõem.
PROPOSTA: dict[str, list[int]] = {
    # --- problemas de entrega -------------------------------------------
    # Nunca chegou. Todos com atraso muito acima da base e nota <= 2.
    "nao_recebido": [15, 31, 69, 64, 91, 58, 74],
    # Chegou, mas fora do prazo.
    "entrega_atrasada": [8, 17, 11, 83, 81],
    # Custo e logística de retirada, não o prazo.
    "frete_e_retirada": [3, 78],

    # --- problemas de produto -------------------------------------------
    # Faltou item do pedido. Atraso NA BASE: não é problema de entrega.
    "quantidade_errada": [0, 53, 84, 95, 47],
    # Veio outro produto, outra cor, outra voltagem.
    "produto_errado": [30, 7, 89, 80],
    # Danificado no transporte ou na embalagem.
    "produto_danificado": [4],
    # Material/acabamento abaixo do esperado.
    "qualidade_abaixo": [21, 33, 50, 71, 59, 96, 76, 93],
    # Não corresponde ao anúncio: foto, réplica, falsificado.
    "divergente_do_anuncio": [38, 70, 28, 62, 55],

    # --- problemas de atendimento e pós-venda ---------------------------
    "atendimento_sem_resposta": [5],
    "cancelamento_reembolso": [9, 22, 56],
    # NF no nome errado trava garantia e troca: problema operacional próprio.
    "nota_fiscal": [61],

    # --- elogios ---------------------------------------------------------
    "elogio_entrega": [14, 6, 13, 18, 24, 12, 20, 19, 27, 77, 25, 41, 43, 46,
                       49, 52, 60, 66, 68, 73, 75, 79, 82, 85, 87, 88, 99,
                       37, 48, 90, 100],
    "elogio_produto": [2, 26, 29, 40, 23, 34, 42, 54, 65, 92, 94, 98],
    "elogio_loja_atendimento": [39, 44, 72, 86, 45, 57, 16, 67],

    # --- sem um assunto único -------------------------------------------
    # Reviews que tratam de mais de um problema na mesma frase. Confirmado
    # pelos exemplos: o tema 35 reúne "nada da cadeira há 30 dias" (não
    # recebido), "pedi lilás e veio roxo" (produto errado) e "o braço
    # quebrou" (qualidade). O que uniu essas reviews foi o PRODUTO ser
    # cadeira, não o problema ser o mesmo — e categoria de produto já é
    # coluna do corpus. Ficam citáveis na busca e FORA da contagem de
    # "principal problema", pela mesma regra dos 886 sem tema: rótulo errado
    # produz número errado.
    "misto": [1, 97, 51, 32, 35, 36, 63],

    # --- sem conteúdo avaliativo ----------------------------------------
    # Recebeu e ainda não usou. Não é elogio nem reclamação.
    "sem_avaliacao_ainda": [10],
}

#: Temas cuja leitura foi ambígua na primeira passada. Revisados com o
#: usuário em 14/11/2026 lendo as reviews de exemplo: 7 viraram `misto`, 4
#: foram confirmados e 2 mudaram (100 era atraso e é elogio de entrega; 61
#: ganhou macro próprio). Mantidos aqui como histórico da revisão.
REVISADOS = {1, 8, 32, 35, 36, 51, 56, 61, 63, 74, 83, 97, 100}

MOTIVO_REVISAR = {
    1: "palavras genéricas (nao/prazo/ainda/hoje) e nota 3,20 — pode ser atraso ou não-recebido",
    8: "datas e 'previsão' com +42 p.p.: pode ser 'atrasou' ou 'nunca chegou'",
    32: "acessórios de celular com nota 2,44 — categoria de produto, não tipo de problema",
    35: "cadeira com 'parafusos, faltando': faltou peça (quantidade) ou veio danificada?",
    36: "cortina com nota 2,91: mistura elogio e reclamação de tecido",
    51: "palavras ilegíveis (nada/hj/família) com +12 p.p. — tema ruidoso, talvez descartar",
    56: "cartão e estorno: é pós-venda ou problema de pagamento à parte?",
    61: "nota fiscal em nome de outra pessoa — é categoria própria, não se encaixa bem",
    63: "toalhas com nota 2,96: mistura 'finas' (qualidade) e 'lindas' (elogio)",
    74: "presente de natal/aniversário com +33 p.p.: o problema é a data, não o produto",
    83: "greve dos caminhoneiros: causa externa de atraso, talvez mereça categoria própria",
    97: "palavras incoerentes (estrelas/pilhas/imaginei) com nota 3,41 — tema ruidoso",
    100: "'demorou pouco' com nota 3,33: é elogio de entrega, não atraso",
}


def montar() -> pd.DataFrame:
    revisao = pd.read_csv(PROCESSED / "temas_para_revisar.csv")

    mapa = {t: macro for macro, temas in PROPOSTA.items() for t in temas}
    faltando = set(revisao.tema) - set(mapa)
    sobrando = set(mapa) - set(revisao.tema)
    if faltando or sobrando:
        raise SystemExit(f"proposta inconsistente: faltando {faltando}, "
                         f"sobrando {sobrando}")

    revisao["macro_proposto"] = revisao.tema.map(mapa)
    revisao["confianca"] = ["revisado" if t in REVISADOS else "alta"
                            for t in revisao.tema]
    revisao["motivo_revisar"] = revisao.tema.map(MOTIVO_REVISAR).fillna("")
    revisao["macro_corrigido"] = ""  # a coluna que ele preenche

    cols = ["tema", "macro_proposto", "macro_corrigido", "confianca",
            "motivo_revisar", "n_docs", "nota_media", "delta_nota",
            "atraso_pct", "delta_atraso_pp", "categoria_dominante",
            "pct_categoria_dominante", "palavras",
            "exemplo_1", "exemplo_2", "exemplo_3"]
    return revisao[cols].sort_values(
        ["macro_proposto", "n_docs"], ascending=[True, False]
    )


def main() -> None:
    revisao = montar()
    destino = PROCESSED / "macro_temas_proposta.csv"
    revisao.to_csv(destino, index=False, encoding="utf-8-sig")

    resumo = (
        revisao.groupby("macro_proposto")
        .agg(temas=("tema", "count"), docs=("n_docs", "sum"),
             nota=("nota_media", "mean"), atraso=("atraso_pct", "mean"))
        .sort_values("docs", ascending=False)
    )
    total = resumo.docs.sum()
    print(f"{len(revisao)} temas -> {len(resumo)} macro-temas  "
          f"({total:,} documentos)\n")
    print(f"{'macro-tema':<26}{'temas':>6}{'docs':>8}{'%':>7}{'nota':>7}{'atraso':>8}")
    print("-" * 62)
    for nome, r in resumo.iterrows():
        print(f"{nome:<26}{r.temas:>6}{int(r.docs):>8,}"
              f"{r.docs / total * 100:>6.1f}%{r.nota:>7.2f}{r.atraso:>7.1f}%")

    n_rev = int((revisao.confianca == "revisado").sum())
    print(f"\n{n_rev} temas revisados com o usuário "
          f"({revisao.loc[revisao.confianca == 'revisado', 'n_docs'].sum():,} docs, "
          f"{revisao.loc[revisao.confianca == 'revisado', 'n_docs'].sum() / total * 100:.1f}%)")
    print(f"gravado em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
