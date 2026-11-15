"""Testes da medida de fidelidade.

Os três casos que importam: resposta ancorada passa, alucinação de sujeito é
pega, e resposta que é só moldura não é confundida com alucinação.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.evaluation.fidelidade import avaliar  # noqa: E402

EVIDENCIA = [
    "O produto chegou com a caixa toda amassada! nenhum cuidado com o produto",
    "Deveriam embalar melhor o produto. A caixa veio toda amassada",
]


def test_resposta_ancorada_tem_taxa_alta():
    f = avaliar("Os clientes relatam a caixa amassada", EVIDENCIA)
    assert f.taxa == 1.0
    assert f.orfaos == []


def test_alucinacao_de_sujeito_e_pega():
    """A falha mais comum de modelo pequeno: inventar o assunto.

    Nenhuma das reviews fala de geladeira, de marca nem de manual. Se a
    resposta fala, ela não veio da evidência — e isso é verificável por código,
    sem juiz de modelo.
    """
    f = avaliar("a geladeira Brastemp veio sem o manual", EVIDENCIA)
    # `veio` ancora de verdade ("A caixa veio toda amassada"), então a taxa não
    # é zero — e não deve ser. O que importa é que o SUJEITO inventado fique
    # entre os órfãos e derrube a taxa para longe do patamar de resposta boa.
    assert f.taxa <= 0.25
    for inventado in ("gelad", "brastemp", "man"):
        assert inventado in f.orfaos


def test_moldura_da_resposta_nao_conta_como_orfa():
    """"Os clientes relatam que..." é moldura, não afirmação sobre o produto.

    Exigir que "relatar" apareça na review penalizaria a resposta bem escrita
    — justamente a que diz de quem é a fala.
    """
    f = avaliar("Os clientes relatam e mencionam problemas", EVIDENCIA)
    assert "relat" not in f.radicais
    assert "client" not in f.radicais


def test_resposta_so_de_moldura_nao_e_infiel():
    """Sem radical de conteúdo, não há o que estar infiel.

    Resposta vazia é problema de utilidade, não de alucinação — e quem pega
    utilidade é a leitura humana, não esta métrica.
    """
    f = avaliar("Os clientes relatam isso", EVIDENCIA)
    assert f.radicais == []
    assert f.taxa == 1.0


def test_a_mesma_rota_lexical_do_indice_e_usada():
    """"atrasada" na resposta e "atraso" na review têm de virar o mesmo radical.

    Se a métrica normalizasse diferente do índice, mediria ancoragem contra um
    alvo deslocado — e acusaria de alucinação a palavra que está lá.
    """
    f = avaliar("a entrega atrasada", ["o atraso na entrega foi grande"])
    assert f.orfaos == []


def test_negacao_nao_e_detectada_e_isso_esta_declarado():
    """Limite conhecido da métrica, fixado em teste para não ser esquecido.

    "chegou" e "não chegou" compartilham o radical. Palavra ancorada não
    garante afirmação correta — a métrica pega sujeito inventado, não relação
    invertida. Quem cobre isso é a leitura das fontes no `:fontes`.
    """
    f = avaliar("o produto chegou rápido", ["o produto não chegou rápido"])
    assert f.taxa == 1.0        # passa, e não deveria dar confiança sozinha


# ---------------------------------------------------------------------------
# Separar INVENÇÃO de ECO DA PERGUNTA e de citação imprecisa. Esta distinção
# é o que torna a métrica interpretável: sem ela, 40% de "invenção" era na
# verdade paráfrase da pergunta e conteúdo de review vista e não citada.
# ---------------------------------------------------------------------------

def test_repetir_a_pergunta_nao_e_invencao():
    """Reformular a pergunta tem origem conhecida e verificável.

    É defeito de utilidade — a resposta que devolve a pergunta não informa
    nada — e não de fidelidade. Misturar os dois faz a métrica acusar de
    alucinação uma frase cuja origem está à vista.
    """
    f = avaliar("Clientes relatam menos unidades do que compraram",
                ["veio só uma peça"],
                pergunta="clientes que receberam menos unidades do que compraram")
    assert f.inventados == []
    assert "unidad" in f.eco_da_pergunta


def test_conteudo_de_review_vista_e_nao_citada_nao_e_invencao():
    """Usar review que o modelo leu e não citou é imprecisão de CITAÇÃO.

    Quem trata isso é o guardrail de citação, que já existe. Contar como
    alucinação seria punir duas vezes o mesmo defeito e esconder o de verdade.
    """
    f = avaliar(
        resposta="Clientes relatam caixa amassada",
        evidencias=["o produto chegou rápido"],          # a citada
        pergunta="",
        vistas=["o produto chegou rápido", "a caixa veio amassada"],
    )
    assert f.inventados == []
    assert "amass" in f.orfaos          # não ancora na citada — e isso aparece


def test_invencao_de_verdade_continua_sendo_pega():
    f = avaliar("Clientes relatam que a geladeira Brastemp queimou",
                ["a caixa veio amassada"],
                pergunta="o que dizem sobre a embalagem?",
                vistas=["a caixa veio amassada"])
    assert "brastemp" in f.inventados
    assert f.taxa_invencao > 0


def test_negacao_fica_fora_do_escopo_da_metrica():
    """A métrica declara que não julga relação; então não pontua negação.

    Antes desta exclusão ela acusava de invenção o "não" de uma frase
    correta — medindo justamente o que admite não medir.
    """
    f = avaliar("Clientes relatam que não receberam o produto",
                ["comprei e nunca chegou"],
                pergunta="quem não recebeu?")
    assert "não" not in f.radicais
    assert "não" not in f.inventados


def test_vazamento_do_exemplo_do_prompt_e_nomeado():
    """Nomear a causa é o que torna o achado acionável.

    Radical que não está na evidência nem na pergunta mas está no exemplo
    few-shot não é o modelo inventando do nada: é o exemplo sendo lido como
    evidência. Foi assim que o defeito de 15/11 foi encontrado.
    """
    from src.generation.responder import TEXTOS_DOS_EXEMPLOS

    assert any("manual" in t for t in TEXTOS_DOS_EXEMPLOS)
    f = avaliar("Clientes relatam que o manual veio em outro idioma",
                ["a caixa veio amassada"],
                pergunta="o que dizem sobre a embalagem?",
                vistas=["a caixa veio amassada"])
    assert "idiom" in f.vindos_do_exemplo
    assert set(f.vindos_do_exemplo) <= set(f.inventados)


def test_exemplo_do_prompt_nao_resume_alem_do_que_mostra():
    """Exemplo que extrapola ensina a extrapolar.

    A resposta de cada exemplo few-shot só pode usar conteúdo que esteja nas
    reviews DAQUELE exemplo. A versão anterior do exemplo 3 terminava em
    "como cor trocada e modelo diferente do anunciado" — conteúdo ausente das
    suas próprias reviews — e era essa parte que reaparecia em resposta real
    sem apoio na evidência.
    """
    from src.evaluation.fidelidade import _radicais
    from src.generation.responder import (
        _EX1_RESPOSTA, _EX1_REVIEWS, _EX3_RESPOSTA, _EX3_REVIEWS,
    )

    for reviews, resposta in ((_EX1_REVIEWS, _EX1_RESPOSTA),
                              (_EX3_REVIEWS, _EX3_RESPOSTA)):
        f = avaliar(resposta, list(reviews))
        assert f.inventados == [], f"exemplo extrapola: {f.inventados}"


def test_toda_entrada_da_moldura_e_um_radical_de_verdade():
    """Lista de exclusão que não casa falha em SILÊNCIO: não quebra, só para
    de excluir.

    `MOLDURA_RESPOSTA` é comparada contra a saída do RSLP, então cada entrada
    tem de ser um radical, não uma palavra. A primeira versão tinha `caso`,
    `comprador`, `principal` e `porque` por extenso — e nenhum casava.
    """
    from src.knowledge_base.limpeza import NEGACOES, pipeline_lexical_bm25
    from src.evaluation.fidelidade import MOLDURA_RESPOSTA

    for entrada in MOLDURA_RESPOSTA:
        if entrada in NEGACOES:
            continue              # negação passa inteira, por decisão da limpeza
        tokens = pipeline_lexical_bm25([entrada])[0]
        assert tokens, f"{entrada!r} é descartada pela rota lexical"
