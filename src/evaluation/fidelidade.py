"""Fidelidade da resposta à evidência citada, medida sem juiz de modelo.

O que isto mede, e o que NÃO mede
---------------------------------
O `cruzamento.md` previa usar RAGAS para *faithfulness* e *answer relevancy*.
RAGAS decompõe a resposta em afirmações e pede a um LLM juiz que diga se cada
uma se sustenta no contexto. Aqui isso não é viável e não por preguiça: não há
chave de API, e um segundo modelo num processo com 0,2 GB de RAM livre é a
mesma causa que já tornou a geração 50x mais lenta. Usar um modelo de 0,5B
como juiz do próprio modelo de 0,5B seria pior que não medir — mediria a
concordância de um modelo fraco consigo mesmo e chamaria isso de fidelidade.

Então mede-se o que **dá para verificar por código**: todo radical de conteúdo
da resposta tem de aparecer em alguma das reviews citadas. É ancoragem
lexical, e é a forma mais comum de alucinação de modelo pequeno — introduzir
um sujeito que não está na evidência (uma marca, um preço, um produto
diferente). Essa, o teste pega.

O que ele **não pega**, declarado:

- **negação e relação.** "o produto chegou rápido" e "o produto não chegou
  rápido" têm quase os mesmos radicais. Palavra ancorada não garante
  afirmação correta.
- **paráfrase legítima.** Responder "demora" onde a review diz "atraso" conta
  como não ancorado, e não é alucinação. Por isso a métrica tem piso útil mas
  não tem teto: 100% não é a meta, e queda brusca é o sinal.

O que cobre a outra metade
--------------------------
*Answer relevancy* — a resposta responde à pergunta? — já é medida de outra
forma neste projeto, e por acerto de **macro-tema** em `calibrar.py`: as
reviews recuperadas pertencem ao tema que a pergunta pedia. Isso é verificável
sem juiz, porque o gabarito é humano.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.knowledge_base.limpeza import pipeline_lexical_bm25  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
REPORTS = RAIZ / "reports"

#: Radicais que a resposta usa para *falar sobre* as reviews, e que por isso
#: não precisam estar nelas. "Os clientes relatam que..." é moldura da
#: resposta, não afirmação sobre o produto — exigir que "relatar" apareça na
#: review penalizaria justamente a resposta bem escrita.
#: ATENÇÃO: todas as entradas são RADICAIS do RSLP, não palavras. A primeira
#: versão tinha `caso`, `comprador`, `principal` e `porque` escritos por
#: extenso, e eles nunca casavam — o radical é `cas`, `compr`, `princip`,
#: `porqu`. Uma lista de exclusão que não casa falha em silêncio: ela não
#: quebra, só deixa de excluir. Há teste que confere cada entrada contra o
#: stemizador para que isso não volte.
MOLDURA_RESPOSTA = frozenset({
    # quem fala: "os clientes relatam", "compradores mencionam"
    "client", "consum", "compr", "relat", "afirm", "diz", "mencion",
    "cit", "reclam", "coment", "avali", "review", "most", "indic", "aparec",
    # quantificação e organização do relato, não conteúdo do relato
    "alg", "mai", "princip", "problem", "cas", "vez", "pesso", "segund",
    # auxiliares: "ter recebido", "foram entregues" — gramática, não conteúdo
    "ter", "est", "sid", "hav",
    # Negação e conjunção são RELAÇÃO, e esta métrica declara no docstring que
    # não julga relação: "chegou" e "não chegou" têm os mesmos radicais.
    # Mantê-las no escopo seria medir o que a métrica admite não medir, e o
    # resultado prático era acusar de invenção o "não" de uma frase correta.
    # Quem cobre relação invertida é a leitura das fontes no `:fontes`.
    "não", "nem", "nunca", "sem", "jamais", "nada", "nenhum", "nenhuma", "mas",
    "poi", "porqu", "após", "ant", "enta", "assim", "tamb",
})


@dataclass
class Fidelidade:
    radicais: list[str]
    ancorados: list[str]
    orfaos: list[str] = field(default_factory=list)
    #: Órfãos que vêm da PERGUNTA. Repetir a pergunta não é inventar: a
    #: palavra tem origem conhecida e verificável. É um defeito de utilidade
    #: (resposta que reformula a pergunta em vez de responder), não de
    #: fidelidade — e separar os dois é o que torna a métrica interpretável.
    eco_da_pergunta: list[str] = field(default_factory=list)
    #: Órfãos sem origem na evidência NEM na pergunta. Estes são os
    #: candidatos reais a alucinação.
    inventados: list[str] = field(default_factory=list)
    #: Subconjunto de `inventados` que vem dos exemplos few-shot do prompt.
    #: Nomear a causa é o que torna o achado acionável: não é o modelo
    #: inventando do nada, é o exemplo sendo lido como evidência.
    vindos_do_exemplo: list[str] = field(default_factory=list)

    @property
    def taxa(self) -> float:
        """Fração dos radicais de conteúdo que aparecem na evidência.

        Sem radical de conteúdo, devolve 1.0: uma resposta que é toda moldura
        ("os clientes relatam isso") não afirma nada sobre o produto e por
        isso não pode estar infiel. Ela é vazia, não é alucinação — e vazia é
        problema de utilidade, que a leitura humana pega.
        """
        return 1.0 if not self.radicais else len(self.ancorados) / len(self.radicais)

    @property
    def taxa_invencao(self) -> float:
        """Fração de radicais sem origem na evidência nem na pergunta.

        **Esta é a métrica de fidelidade propriamente dita.** `taxa` mistura
        alucinação com paráfrase e eco da pergunta; `taxa_invencao` isola o
        que não tem origem nenhuma. O alvo é zero, e aqui zero é atingível —
        ao contrário de 100% de ancoragem, que penalizaria o sinônimo.
        """
        return 0.0 if not self.radicais else len(self.inventados) / len(self.radicais)

    def __str__(self) -> str:
        linha = (f"ancoragem {self.taxa:.0%} "
                 f"({len(self.ancorados)}/{len(self.radicais)} radicais), "
                 f"invenção {self.taxa_invencao:.0%}")
        if self.inventados:
            linha += f" | INVENTADOS: {self.inventados}"
        if self.vindos_do_exemplo:
            linha += f" | DO EXEMPLO DO PROMPT: {self.vindos_do_exemplo}"
        if self.eco_da_pergunta:
            linha += f" | eco da pergunta: {self.eco_da_pergunta}"
        return linha


@lru_cache(maxsize=1)
def _radicais_do_prompt() -> frozenset[str]:
    """Radicais que existem nos exemplos few-shot do prompt de geração.

    Serve para **nomear a causa** de um radical inventado, não para barrar
    resposta. A tentativa de usar isto como guardrail de produção falhou de
    forma instrutiva e está registrada em `decisoes.md`: a moldura de relato
    que os exemplos ensinam ("os clientes relatam") é, por construção,
    vocabulário que review nenhuma usa — `relat` tem IDF 7,78 no corpus,
    mais raro que `fon` (6,51). Logo moldura e contaminação não são
    separáveis por raridade, e o guardrail abstinha 7 de 8 respostas.

    Aqui, na camada de avaliação, um falso positivo não custa resposta
    nenhuma — custa uma linha de diagnóstico a mais. É o lugar certo para um
    sinal que é informativo e não é confiável.
    """
    from src.generation.responder import TEXTOS_DOS_EXEMPLOS

    radicais: set[str] = set()
    for tokens in pipeline_lexical_bm25(list(TEXTOS_DOS_EXEMPLOS)):
        radicais |= set(tokens)
    return frozenset(radicais)


def _radicais(texto: str) -> list[str]:
    """Radicais de conteúdo, pela mesma rota lexical que o BM25 usa.

    Reusar a rota do índice não é economia de código: é o que garante que
    "atrasada" na resposta e "atraso" na review virem o mesmo radical. Duas
    normalizações diferentes mediriam ancoragem contra um alvo deslocado.
    """
    return pipeline_lexical_bm25([texto])[0]


def avaliar(resposta: str, evidencias: list[str],
            pergunta: str = "", vistas: list[str] | None = None) -> Fidelidade:
    """Confere a origem de cada radical de conteúdo da resposta.

    Três destinos possíveis: está na evidência (ancorado), está na pergunta
    (eco — origem conhecida, não é invenção) ou não está em lugar nenhum
    (candidato a alucinação).

    `evidencias` são as reviews **citadas** e medem a ancoragem — se a citação
    sustenta a frase. `vistas` são todas as reviews que foram ao prompt, e é
    contra elas que a **invenção** é medida: usar conteúdo de uma review que o
    modelo leu e não citou é imprecisão de citação, que o guardrail já trata,
    e não invenção. Confundir as duas acusaria de alucinação uma afirmação
    que tem origem verificável. Sem `vistas`, cai para `evidencias`.
    """
    def radicais_de(textos: list[str]) -> set[str]:
        saida: set[str] = set()
        for t in pipeline_lexical_bm25(textos):
            saida |= set(t)
        return saida

    da_evidencia = radicais_de(evidencias)
    do_contexto = radicais_de(vistas) if vistas else da_evidencia
    da_pergunta = set(_radicais(pergunta)) if pergunta else set()

    radicais = [r for r in dict.fromkeys(_radicais(resposta))
                if r not in MOLDURA_RESPOSTA]
    ancorados = [r for r in radicais if r in da_evidencia]
    orfaos = [r for r in radicais if r not in da_evidencia]
    inventados = [r for r in orfaos
                  if r not in do_contexto and r not in da_pergunta]
    return Fidelidade(
        radicais=radicais,
        ancorados=ancorados,
        orfaos=orfaos,
        eco_da_pergunta=[r for r in orfaos
                         if r not in do_contexto and r in da_pergunta],
        inventados=inventados,
        vindos_do_exemplo=[r for r in inventados if r in _radicais_do_prompt()],
    )


# ----------------------------------------------------------------------
PERGUNTAS = [
    "o que dizem os clientes sobre a caixa amassada?",
    "o que dizem os clientes que não receberam o produto?",
    "reclamações sobre os Correios",
    "clientes que receberam menos unidades do que compraram",
    "veio um produto diferente do que eu pedi",
    "pedi o cancelamento da compra",
    "tentei entrar em contato e nunca responderam",
    "o que os clientes dizem sobre demora na entrega?",
]


def main() -> None:
    import pandas as pd

    from src.generation.responder import Gerador

    print("FIDELIDADE DA GERAÇÃO — ancoragem lexical na evidência citada")
    print("Não é RAGAS: não há juiz de modelo. Ver o docstring para o limite.\n")

    g = Gerador()
    linhas = []
    for pergunta in PERGUNTAS:
        r = g.responder(pergunta)
        if not r.respondeu:
            print(f"  ABSTEVE ({r.abstencao_por}) | {pergunta}")
            linhas.append({"pergunta": pergunta, "respondeu": False,
                           "suficiencia": r.suficiencia, "taxa": None,
                           "taxa_invencao": None, "inventados": "",
                           "eco_da_pergunta": "",
                           "latencia_ms": round(r.latencia_ms)})
            continue

        mapa = g.busca.mapa
        citadas = mapa[mapa.review_id.isin(r.citacoes)].texto.tolist()
        vistas = mapa[mapa.review_id.isin(r.evidencias)].texto.tolist()
        f = avaliar(r.texto, citadas, pergunta, vistas)
        marca = "INVENTOU" if f.inventados else "OK      "
        print(f"  {marca} {f}")
        print(f"        pergunta: {pergunta}")
        print(f'        resposta: "{r.texto}"')
        print(f"        evidência: {len(citadas)} reviews, suficiência {r.suficiencia}")
        linhas.append({"pergunta": pergunta, "respondeu": True,
                       "suficiencia": r.suficiencia, "taxa": round(f.taxa, 3),
                       "taxa_invencao": round(f.taxa_invencao, 3),
                       "inventados": " ".join(f.inventados),
                       "eco_da_pergunta": " ".join(f.eco_da_pergunta),
                       "latencia_ms": round(r.latencia_ms)})

    df = pd.DataFrame(linhas)
    respondidas = df[df.respondeu]
    print("\n" + "=" * 74)
    print(f"respondidas: {len(respondidas)}/{len(df)}")
    if len(respondidas):
        com_invencao = int((respondidas.taxa_invencao > 0).sum())
        print(f"ancoragem mediana:  {respondidas.taxa.median():.0%}")
        print(f"ancoragem mínima:   {respondidas.taxa.min():.0%}")
        print(f"INVENÇÃO mediana:   {respondidas.taxa_invencao.median():.0%}"
              f"   <- esta é a métrica de fidelidade")
        print(f"respostas com algum radical inventado: "
              f"{com_invencao}/{len(respondidas)}")
        print(f"latência mediana:   {respondidas.latencia_ms.median():.0f} ms")
    print("custo por pergunta: R$ 0,00 — embedding e geração rodam local")

    REPORTS.mkdir(exist_ok=True)
    destino = REPORTS / "fidelidade.csv"
    df.to_csv(destino, index=False, encoding="utf-8")
    print(f"\ngravado em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
