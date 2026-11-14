"""O assistente completo: roteia a pergunta e devolve a resposta certa.

Decisão de 12/11: **workflow fixo, não agente**. A alternativa era um agente
ReAct com ferramentas, que escolheria sozinho entre contar e buscar. Três
razões para não: o caminho é previsível e testável, a decisão é explicável
numa banca, e um modelo de 1,5B escolhendo ferramenta erra muito. A aula 04 de
IA Generativa mostra o loop ReAct justamente para deixar claro quando ele
compensa — e aqui não compensa, porque só existem duas rotas.

As duas rotas, e por que a distinção é obrigatória:

    pergunta de FREQUÊNCIA   -> camada de contagem (agregação por macro-tema)
    pergunta de EVIDÊNCIA    -> RAG (busca híbrida + geração citada)

Mandar "quais os principais problemas?" para o RAG devolveria as 5 reviews
mais parecidas com a palavra "problema" — não os problemas principais. E
mandar "o que dizem sobre a caixa amassada?" para a contagem devolveria um
número onde se pediu depoimento. São perguntas de naturezas diferentes e a
mesma peça não serve para as duas.

O roteador é de regra, e isso é deliberado: regra erra de forma previsível e
auditável. Quando ele não tem certeza, a pergunta vai para o RAG, porque o RAG
tem o portão de suficiência atrás dele — errar para o lado que sabe abster é
melhor que errar para o lado que afirma número.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generation.agregacao import ROTULOS, Agregador, RespostaAgregada  # noqa: E402
from src.generation.responder import Gerador, Resposta  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]

#: Pede CONTAGEM: quer saber o que é mais frequente, não o que alguém disse.
PADROES_FREQUENCIA = [
    r"\bprincipais?\b",
    # `mais <verbo>` cobre "o que mais reclamam" e também "do que os clientes
    # mais elogiam", onde há palavras entre o "que" e o verbo.
    r"\bmais (reclam|elogi|gost|cita|relat|aparec|comum|frequent)",
    r"\bquantos?\b", r"\bquantas?\b", r"\bqual (o |a )?(percentual|propor[çc])",
    r"\branking\b", r"\bdistribui[çc]", r"\bresumo geral\b",
    r"\bo que (mais )?(reclamam|elogiam)\b", r"\btop \d",
    r"\bmaiores (problemas|reclama)", r"\bvis[ãa]o geral\b",
]

#: Pede SÉRIE TEMPORAL: quer saber se mudou ao longo do tempo.
PADROES_TEMPORAL = [
    r"\b(piorou|melhorou|aumentou|diminuiu|caiu|subiu)\b",
    r"\bevolu[çc]", r"\bao longo (do tempo|dos meses)", r"\bpor m[êe]s\b",
    r"\btend[êe]ncia\b", r"\bem algum m[êe]s\b", r"\bqual m[êe]s\b",
]

#: Pede ELOGIO em vez de problema.
PADROES_ELOGIO = [r"\belogi", r"\bpontos? (positivos?|fortes?)\b",
                  r"\bo que (gostam|mais gostam)\b", r"\bsatisfeit"]

#: Filtros extraíveis da própria pergunta (item opcional do enunciado).
PADROES_ATRASADAS = [r"\batrasad", r"\bfora do prazo\b", r"\bcom atraso\b",
                     r"\bentrega atrasada\b"]

_UFS = {"AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
        "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
        "SP", "SE", "TO"}

#: Palavras que apontam um macro-tema específico, para a rota temporal saber
#: QUAL série traçar. Derivado dos rótulos, mais sinônimos de uso comum.
GATILHOS_MACRO = {
    "nao_recebido": ["não receb", "nao receb", "não cheg", "nao cheg",
                     "nunca cheg", "não entreg", "nao entreg", "extravi"],
    "entrega_atrasada": ["atras", "demor", "fora do prazo", "prazo"],
    "produto_danificado": ["danific", "quebrad", "amassad", "violad", "embalag"],
    "quantidade_errada": ["quantidade", "falt", "unidade", "incomplet"],
    "produto_errado": ["errad", "diferente do que", "outro produto", "voltagem"],
    "qualidade_abaixo": ["qualidade", "material", "frágil", "fragil"],
    "divergente_do_anuncio": ["foto", "anúncio", "anuncio", "réplica", "replica",
                              "falsific", "propaganda"],
    "cancelamento_reembolso": ["cancel", "reembols", "estorn", "dinheiro de volta"],
    "atendimento_sem_resposta": ["atendimento", "contato", "sac", "resposta"],
    "frete_e_retirada": ["frete", "correio", "agência", "agencia", "retirar"],
    "nota_fiscal": ["nota fiscal", "nf"],
}


@dataclass
class Rota:
    tipo: str                 # "frequencia" | "temporal" | "evidencia"
    motivo: str
    elogios: bool = False
    macro: str | None = None
    estado: str | None = None
    so_atrasadas: bool = False

    def __str__(self) -> str:
        extras = [f"elogios={self.elogios}" if self.elogios else "",
                  f"macro={self.macro}" if self.macro else "",
                  f"UF={self.estado}" if self.estado else "",
                  "só atrasadas" if self.so_atrasadas else ""]
        extras = ", ".join(e for e in extras if e)
        return f"{self.tipo}{f' [{extras}]' if extras else ''} — {self.motivo}"


def _casa(pergunta: str, padroes: list[str]) -> str | None:
    for p in padroes:
        if re.search(p, pergunta, flags=re.IGNORECASE):
            return p
    return None


def rotear(pergunta: str) -> Rota:
    """Classifica a pergunta. Na dúvida, manda para o RAG."""
    p = pergunta.lower()

    # Filtros valem para qualquer rota.
    estado = next((uf for uf in _UFS
                   if re.search(rf"\b{uf}\b", pergunta)), None)
    atrasadas = bool(_casa(p, PADROES_ATRASADAS))
    elogios = bool(_casa(p, PADROES_ELOGIO))

    macro = None
    for nome, gatilhos in GATILHOS_MACRO.items():
        if any(g in p for g in gatilhos):
            macro = nome
            break

    # Temporal primeiro: "a reclamação de atraso piorou?" é temporal, e também
    # casaria com frequência se a ordem fosse outra.
    if (achou := _casa(p, PADROES_TEMPORAL)):
        return Rota("temporal", f"padrão temporal {achou!r}",
                    elogios=elogios, macro=macro or "nao_recebido",
                    estado=estado, so_atrasadas=atrasadas)

    if (achou := _casa(p, PADROES_FREQUENCIA)):
        return Rota("frequencia", f"padrão de frequência {achou!r}",
                    elogios=elogios, macro=macro, estado=estado,
                    so_atrasadas=atrasadas)

    return Rota("evidencia", "nenhum padrão de contagem — vai para o RAG, "
                             "que tem o portão de suficiência atrás",
                elogios=elogios, macro=macro, estado=estado,
                so_atrasadas=atrasadas)


class Assistente:
    """A porta única do sistema. Carrega as duas rotas uma vez."""

    def __init__(self, *, carregar_llm: bool = True) -> None:
        self.agregador = Agregador()
        self.gerador = Gerador() if carregar_llm else None

    def perguntar(self, pergunta: str) -> tuple[Rota, Resposta | RespostaAgregada]:
        rota = rotear(pergunta)

        if rota.tipo == "temporal":
            return rota, self.agregador.por_periodo(pergunta, rota.macro)

        if rota.tipo == "frequencia":
            return rota, self.agregador.principais(
                pergunta, elogios=rota.elogios, estado=rota.estado,
                so_atrasadas=rota.so_atrasadas)

        if self.gerador is None:
            raise RuntimeError("rota de evidência exige o LLM carregado")
        return rota, self.gerador.responder(pergunta)


PERGUNTAS_DEMO = [
    "quais os principais problemas relatados pelos clientes?",
    "do que os clientes mais elogiam?",
    "a reclamação de não recebimento piorou em algum mês?",
    "entre as entregas atrasadas, quais os maiores problemas?",
    "quais os principais problemas em SP?",
    "o que dizem os clientes sobre a caixa amassada?",
    "quem não recebeu o produto, o que relatou?",
    "qual a política de troca e devolução da Olist?",
]


def main() -> None:
    print("ROTEAMENTO (sem carregar o LLM)\n")
    for p in PERGUNTAS_DEMO:
        print(f"  {rotear(p)}\n    <- {p}")

    print("\n" + "=" * 76)
    print("EXECUÇÃO COMPLETA\n")
    a = Assistente()
    for p in PERGUNTAS_DEMO:
        rota, resp = a.perguntar(p)
        print("=" * 76)
        print(f"ROTA: {rota}")
        print(resp)
        print()


if __name__ == "__main__":
    main()
