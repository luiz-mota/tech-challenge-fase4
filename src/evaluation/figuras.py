"""Gera as duas figuras do README e do vídeo.

Por que só duas, e não um painel: cada figura responde **uma** pergunta que o
texto sozinho não responde bem.

  1. `temas_problema.png` — qual a distribuição dos problemas. É a resposta da
     rota de frequência, com o denominador declarado dentro da própria figura:
     a fatia sai sobre os documentos classificados como problema, nunca sobre
     as 99.224 avaliações da base.

  2. `tema_vs_nota.png` — por que a camada de temas existe. Esta é a figura que
     mais rende, porque a validação é **independente do texto**: `atraso_dias`
     vem das datas do pedido, não do que o cliente escreveu. Se os temas
     formados a partir do texto separam grupos com taxa de atraso muito
     diferente, eles capturaram assunto de verdade — e não apenas palavras
     parecidas. A nota, que é o metadado óbvio, não separa esses grupos.

Decisões de visualização, para não precisar defendê-las de improviso:

  - **Uma série, uma cor.** Nenhuma das duas figuras compara séries, então
    colorir barra por barra seria cor sem significado — o olho procura um
    agrupamento que não existe. A cor única deixa o comprimento da barra ser a
    única coisa que varia, que é o que se quer ler.
  - **Barra horizontal, não pizza nem barra vertical.** São 11 categorias com
    nome longo; na vertical o rótulo vira diagonal e na pizza não se compara
    fatia vizinha. Comprimento a partir de uma linha de base comum é a
    comparação mais fácil de fazer a olho.
  - **Rótulo direto em vez de eixo.** Figura estática não tem tooltip, então o
    número vai na barra e a grade fica recessiva.
  - **Linha de referência na figura 2.** Sem a média do corpus, "41%" não
    significa nada. A linha é o que transforma o número em comparação.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # sem display: roda em CI e em servidor
import matplotlib.patheffects as pe       # noqa: E402
import matplotlib.pyplot as plt           # noqa: E402
import pandas as pd                       # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.generation.agregacao import ROTULOS  # noqa: E402

PROCESSED = RAIZ / "data" / "processed"
FIGURAS = RAIZ / "reports" / "figuras"

#: Paleta validada: uma série, um tom. Azul passa contraste >= 3:1 na superfície
#: clara, que é o mínimo para marca colorida sem depender de rótulo.
AZUL = "#2a78d6"
CINZA_TEXTO = "#52514e"
CINZA_GRADE = "#dcdbd7"
TINTA = "#0b0b0b"
SUPERFICIE = "#fcfcfb"


def _estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": CINZA_GRADE,
        "axes.labelcolor": CINZA_TEXTO,
        "text.color": TINTA,
        "xtick.color": CINZA_TEXTO,
        "ytick.color": TINTA,
        # Em barra horizontal a linha de base é x=0, ou seja a espinha
        # ESQUERDA. É a única que ajuda a ler, então é a única que fica.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.bottom": False,
        "savefig.facecolor": SUPERFICIE,
        "savefig.bbox": "tight",
        "savefig.dpi": 160,
    })


def _pct(x: float, casas: int = 1) -> str:
    """Decimal com vírgula. O público é brasileiro; ponto decimal distrai."""
    return f"{x:.{casas}f}%".replace(".", ",")


def _milhar(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _titulo(fig, ax, titulo: str, subtitulo: str, rodape: str) -> None:
    """Título, subtítulo e fonte posicionados sem colidir.

    `ax.set_title` com `pad` grande empurra o título mas não reserva espaço
    para uma segunda linha, então as duas se sobrepõem. Desenhar as duas como
    texto em coordenadas do eixo resolve, e mantém o alinhamento à esquerda.
    """
    ax.text(0, 1.16, titulo, transform=ax.transAxes, fontsize=15,
            fontweight="bold", color=TINTA, va="bottom")
    ax.text(0, 1.055, subtitulo, transform=ax.transAxes, fontsize=9.5,
            color=CINZA_TEXTO, va="bottom")
    fig.text(0.005, 0.005, rodape, fontsize=8.5, color=CINZA_TEXTO,
             va="bottom")


def _carregar() -> pd.DataFrame:
    corpus = pd.read_parquet(PROCESSED / "corpus.parquet")
    temas = pd.read_parquet(PROCESSED / "corpus_temas.parquet")
    d = corpus.merge(temas[["review_id", "tema", "macro",
                            "conta_como_problema"]], on="review_id",
                     how="left")
    d["macro"] = d.macro.fillna("sem_tema")
    d["rotulo"] = d.macro.map(ROTULOS).fillna(d.macro)
    return d


# ---------------------------------------------------------------------------
def figura_temas(d: pd.DataFrame) -> Path:
    """Distribuição dos macro-temas de problema, com o denominador na figura."""
    prob = d[d.conta_como_problema.fillna(False)]
    g = (prob.groupby("rotulo").size().sort_values(ascending=True))
    total = len(prob)

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.barh(g.index, g.values, color=AZUL, height=0.70)

    for nome, valor in g.items():
        ax.text(valor + total * 0.006, nome,
                f"{_milhar(valor)}   {_pct(valor / total * 100)}",
                va="center", ha="left", fontsize=9.5, color=CINZA_TEXTO)

    ax.set_xlim(0, g.max() * 1.26)
    ax.set_xticks([])
    ax.xaxis.grid(False)
    _titulo(
        fig, ax,
        "Do que os clientes reclamam",
        f"{_milhar(total)} avaliações com texto classificadas como problema"
        " · a fatia é sobre esse total, não sobre a base inteira",
        "Fonte: Olist, 39.388 avaliações com texto (2016–2018). "
        "Tema atribuído por BERTopic e agrupado à mão em 16 macro-temas.",
    )

    saida = FIGURAS / "temas_problema.png"
    fig.savefig(saida)
    plt.close(fig)
    return saida


def figura_tema_vs_nota(d: pd.DataFrame) -> Path:
    """A validação independente do texto: o tema separa, a nota não."""
    prob = d[d.conta_como_problema.fillna(False)]
    g = (prob.groupby("rotulo")
             .agg(n=("review_id", "size"),
                  atraso=("entrega_atrasada", lambda s: s.fillna(False).mean() * 100),
                  nota=("review_score", "mean"))
             .sort_values("atraso"))
    base = d.entrega_atrasada.fillna(False).mean() * 100

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.barh(g.index, g.atraso, color=AZUL, height=0.70)

    # A linha de referência cruza o rótulo das barras curtas ("6,6%"). Mandar a
    # linha para trás por `zorder` não resolve: ela reaparece nos vãos entre as
    # letras. O que separa de verdade é um halo da cor da superfície em volta
    # do texto — correção geométrica, não de ordem de desenho.
    halo = [pe.withStroke(linewidth=3.5, foreground=SUPERFICIE)]

    ax.axvline(base, color=TINTA, linewidth=1.3, linestyle=(0, (4, 3)),
               zorder=0.5)
    ax.text(base + 0.7, len(g) - 0.3, f"média do corpus: {_pct(base)}",
            fontsize=9, color=TINTA, va="center", zorder=4,
            path_effects=halo)

    for i, (_, r) in enumerate(g.iterrows()):
        ax.text(r.atraso + 1.0, i,
                f"{_pct(r.atraso)}      nota média "
                + f"{r.nota:.1f}".replace(".", ","),
                va="center", ha="left", fontsize=9.5, color=CINZA_TEXTO,
                zorder=4, path_effects=halo)

    ax.set_xlim(0, g.atraso.max() * 1.62)
    ax.set_xticks([])
    _titulo(
        fig, ax,
        "O tema distingue o que a nota não distingue",
        "% de pedidos que de fato atrasaram, por macro-tema · o atraso vem "
        "das datas do pedido, não do texto",
        '"Produto não recebido" tem 41,2% de atraso real e nota 1,8; '
        '"quantidade errada" tem 4,1% e nota 2,0. A nota é praticamente a '
        "mesma — o tema é que separa a causa.",
    )

    saida = FIGURAS / "tema_vs_nota.png"
    fig.savefig(saida)
    plt.close(fig)
    return saida


def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    _estilo()
    d = _carregar()
    for f in (figura_temas(d), figura_tema_vs_nota(d)):
        print(f"gravado: {f.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
