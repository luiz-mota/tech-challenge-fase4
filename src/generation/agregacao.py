"""Respostas de frequência: contagem sobre macro-temas, com número honesto.

Por que esta camada existe separada do RAG: a busca devolve as reviews mais
**parecidas** com a pergunta, não as mais **frequentes**. Perguntar "quais os
principais problemas?" ao RAG devolve 5 reviews parecidas com a palavra
"problema" — não os problemas principais. Contar exige agregar antes.

As três regras de honestidade da contagem, e cada uma impede um erro
específico:

  1. **O denominador é declarado.** A porcentagem sai sobre os 13.166
     documentos classificados como problema, não sobre as 99.224 avaliações
     nem sobre os 39.388 documentos. Dizer "32% dos clientes reclamam de
     entrega" seria falso; "32% dos relatos de problema são de entrega" é
     verdade.

  2. **O viés de quem escreve é declarado junto.** Só escreve texto quem está
     nos extremos: 77,3% das notas 1 escreveram, contra 32,8% das notas 4.
     Nota 1 é 11,5% da base e 21,9% do corpus. Então a contagem descreve
     **quem escreveu**, nunca **quem comprou**, e a resposta diz isso.

  3. **O que não foi classificado fica visível.** `misto`, `sem_tema` e
     `sem_avaliacao_ainda` aparecem no rodapé, não são escondidos no
     denominador. São 11,5% do corpus e omiti-los inflaria todas as fatias.

Cada macro-tema vem com reviews de exemplo citáveis por `review_id`, para que
a resposta de frequência também seja auditável — o número e a evidência.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.knowledge_base.aplicar_temas import ELOGIOS, FORA_DA_CONTAGEM  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"

#: Quantas reviews de exemplo acompanham cada macro-tema na resposta.
N_EXEMPLOS_POR_TEMA = 2

#: Nome legível de cada macro-tema. O slug é bom para código e ruim para
#: resposta ao usuário.
ROTULOS = {
    "nao_recebido": "produto não recebido",
    "entrega_atrasada": "entrega fora do prazo",
    "frete_e_retirada": "frete e retirada na agência",
    "quantidade_errada": "quantidade errada ou item faltando",
    "produto_errado": "produto, cor ou voltagem errada",
    "produto_danificado": "produto danificado ou embalagem violada",
    "qualidade_abaixo": "qualidade abaixo do esperado",
    "divergente_do_anuncio": "divergente do anúncio (foto, réplica)",
    "atendimento_sem_resposta": "atendimento sem resposta",
    "cancelamento_reembolso": "cancelamento e reembolso",
    "nota_fiscal": "nota fiscal incorreta",
    "elogio_entrega": "elogio à entrega",
    "elogio_produto": "elogio ao produto",
    "elogio_loja_atendimento": "elogio à loja e ao atendimento",
    "misto": "vários assuntos na mesma avaliação",
    "sem_avaliacao_ainda": "recebeu e ainda não avaliou",
    "sem_tema": "sem padrão recorrente",
}

#: Aviso de viés, anexado a toda resposta de frequência. Não é opcional: sem
#: ele o número convida à leitura errada.
AVISO_VIES = (
    "Estes números descrevem quem escreveu avaliação, não quem comprou. "
    "Escrevem sobretudo clientes nos extremos: 77,3% das avaliações nota 1 "
    "têm texto, contra 32,8% das nota 4."
)


def num(n: float | int, casas: int = 0) -> str:
    """Número no formato brasileiro: ponto no milhar, vírgula no decimal.

    O padrão do Python é o inglês (`2,249` e `17.1`), e num relatório em
    português isso não é só feio: `2,249` lê-se como "dois vírgula duzentos e
    quarenta e nove". Quem apresenta o número para um gestor brasileiro não
    pode deixar a formatação contradizer a língua do texto ao lado.
    """
    return f"{n:,.{casas}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def pct(x: float, casas: int = 1) -> str:
    return f"{num(x, casas)}%"


@dataclass
class RespostaAgregada:
    pergunta: str
    texto: str
    tabela: pd.DataFrame
    denominador: int
    rotulo_denominador: str
    citacoes: list[str] = field(default_factory=list)
    aviso: str = AVISO_VIES

    def __str__(self) -> str:
        linhas = [f'PERGUNTA: "{self.pergunta}"', "", self.texto, ""]
        linhas.append(f"  {'tema':<42}{'relatos':>9}{'%':>7}{'nota':>7}{'atraso':>8}")
        linhas.append("  " + "-" * 73)
        for _, r in self.tabela.iterrows():
            linhas.append(f"  {r.rotulo:<42}{num(r.docs):>9}{pct(r.pct):>7}"
                          f"{num(r.nota, 2):>7}{pct(r.atraso):>8}")
        linhas += ["", f"  base: {num(self.denominador)} {self.rotulo_denominador}",
                   f"  {self.aviso}"]
        if self.citacoes:
            linhas.append(f"  evidências: {', '.join(c[:8] for c in self.citacoes)}")
        return "\n".join(linhas)


class Agregador:
    def __init__(self) -> None:
        corpus = pd.read_parquet(PROCESSED / "corpus.parquet")
        temas = pd.read_parquet(PROCESSED / "corpus_temas.parquet")
        self.df = corpus.merge(temas, on="review_id", how="left")
        self.df["macro"] = self.df.macro.fillna("sem_tema")
        self.df["rotulo"] = self.df.macro.map(ROTULOS).fillna(self.df.macro)

    # ------------------------------------------------------------------
    def _filtrar(self, categoria: str | None, estado: str | None,
                 ano_mes: str | None, so_atrasadas: bool) -> tuple[pd.DataFrame, list[str]]:
        """Recuperação filtrada (item opcional do enunciado)."""
        d, descr = self.df, []
        if categoria:
            d = d[d.categoria == categoria]
            descr.append(f"categoria {categoria}")
        if estado:
            d = d[d.customer_state == estado]
            descr.append(f"estado {estado}")
        if ano_mes:
            d = d[d.ano_mes == ano_mes]
            descr.append(f"mês {ano_mes}")
        if so_atrasadas:
            d = d[d.entrega_atrasada.fillna(False)]
            descr.append("só entregas atrasadas")
        return d, descr

    def _exemplos(self, sub: pd.DataFrame) -> list[str]:
        """Reviews mais longas do tema: texto longo descreve, texto curto não."""
        return sub.nlargest(N_EXEMPLOS_POR_TEMA, "n_palavras").review_id.tolist()

    # ------------------------------------------------------------------
    def principais(self, pergunta: str, *, elogios: bool = False, top: int = 6,
                   categoria: str | None = None, estado: str | None = None,
                   ano_mes: str | None = None,
                   so_atrasadas: bool = False) -> RespostaAgregada:
        """Ranking dos macro-temas, de problema (padrão) ou de elogio."""
        d, descr = self._filtrar(categoria, estado, ano_mes, so_atrasadas)

        if elogios:
            alvo = d[d.macro.isin(ELOGIOS)]
            rotulo_den = "avaliações com elogio identificado"
        else:
            alvo = d[~d.macro.isin(ELOGIOS | FORA_DA_CONTAGEM)]
            rotulo_den = "avaliações com problema identificado"
        if descr:
            rotulo_den += " (" + ", ".join(descr) + ")"

        den = len(alvo)
        if den == 0:
            return RespostaAgregada(
                pergunta=pergunta,
                texto="Não há avaliações suficientes com esse filtro para contar.",
                tabela=pd.DataFrame(columns=["rotulo", "docs", "pct", "nota", "atraso"]),
                denominador=0, rotulo_denominador=rotulo_den, citacoes=[])

        tabela = (
            alvo.groupby(["macro", "rotulo"])
            .agg(docs=("review_id", "count"), nota=("review_score", "mean"),
                 atraso=("entrega_atrasada", "mean"))
            .reset_index().sort_values("docs", ascending=False).head(top)
        )
        tabela["pct"] = tabela.docs / den * 100
        tabela["atraso"] = tabela.atraso * 100

        citacoes = []
        for macro in tabela.macro:
            citacoes += self._exemplos(alvo[alvo.macro == macro])

        primeiro = tabela.iloc[0]
        tipo = "elogio" if elogios else "problema"
        texto = (
            f"O {tipo} mais relatado é **{primeiro.rotulo}**, com "
            f"{num(primeiro.docs)} relatos ({pct(primeiro.pct)} das {num(den)} "
            f"avaliações com {tipo} identificado). "
        )
        if not elogios:
            entrega = tabela[tabela.macro.isin(
                {"nao_recebido", "entrega_atrasada", "frete_e_retirada"})]
            if len(entrega):
                texto += (f"Somando os temas de entrega, são "
                          f"{num(entrega.docs.sum())} "
                          f"relatos ({pct(entrega.pct.sum())}). ")

        fora = d[d.macro.isin(FORA_DA_CONTAGEM)]
        texto += (f"Ficam fora desta contagem {num(len(fora))} avaliações "
                  f"({pct(len(fora) / len(d) * 100)} do corpus) que tratam de vários "
                  f"assuntos ao mesmo tempo, não têm padrão recorrente ou ainda não "
                  f"foram avaliadas.")

        return RespostaAgregada(
            pergunta=pergunta, texto=texto,
            tabela=tabela[["rotulo", "docs", "pct", "nota", "atraso"]],
            denominador=den, rotulo_denominador=rotulo_den, citacoes=citacoes)

    def por_periodo(self, pergunta: str, macro: str) -> RespostaAgregada:
        """Evolução mensal de um macro-tema, em % do total do mês.

        Em % e não em contagem absoluta de propósito: o volume de avaliações
        do Olist cresce ao longo do período, e contagem crua mostraria
        crescimento em tudo.
        """
        d = self.df
        do_tema = d[d.macro == macro]
        por_mes = (
            d.groupby("ano_mes").agg(total=("review_id", "count")).join(
                do_tema.groupby("ano_mes").agg(
                    docs=("review_id", "count"),
                    nota=("review_score", "mean"),
                    atraso=("entrega_atrasada", "mean"),
                )
            ).reset_index()
        )
        por_mes[["docs", "nota", "atraso"]] = por_mes[["docs", "nota", "atraso"]].fillna(0)
        # Meses com menos de 100 avaliações saem: com base pequena, uma única
        # review move a porcentagem vários pontos e o "pico" vira artefato.
        por_mes = por_mes[por_mes.total >= 100].copy()
        por_mes["pct"] = por_mes.docs / por_mes.total * 100
        por_mes["atraso"] = por_mes.atraso * 100
        pico = por_mes.loc[por_mes.pct.idxmax()]

        tabela = por_mes.rename(columns={"ano_mes": "rotulo"})
        tabela["docs"] = tabela.docs.astype(int)

        return RespostaAgregada(
            pergunta=pergunta,
            texto=(f"O tema **{ROTULOS.get(macro, macro)}** teve pico em "
                   f"{pico.ano_mes}, com {pct(pico.pct)} das avaliações do mês "
                   f"({num(int(pico.docs))} de {num(int(pico.total))})."),
            tabela=tabela[["rotulo", "docs", "pct", "nota", "atraso"]],
            denominador=int(por_mes.total.sum()),
            rotulo_denominador="avaliações em meses com ao menos 100",
            citacoes=self._exemplos(d[(d.macro == macro) & (d.ano_mes == pico.ano_mes)]))


def main() -> None:
    a = Agregador()
    print("=" * 76)
    print(a.principais("quais os principais problemas relatados pelos clientes?"))
    print("\n" + "=" * 76)
    print(a.principais("e entre quem recebeu com atraso, o que mais reclamam?",
                       so_atrasadas=True))
    print("\n" + "=" * 76)
    print(a.principais("do que os clientes mais elogiam?", elogios=True, top=3))
    print("\n" + "=" * 76)
    print(a.por_periodo("a reclamação de não recebimento piorou em algum mês?",
                        "nao_recebido"))


if __name__ == "__main__":
    main()
