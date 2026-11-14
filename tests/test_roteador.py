"""Testes do roteador.

O roteador é de regra, não de modelo — então ele é testável de forma exata, e
é onde os testes rendem mais. Um erro de rota não aparece como exceção: a
pergunta de contagem vai para o RAG e volta com 5 depoimentos onde se pediu
número. Falha silenciosa é o que teste pega.

Não precisa do LLM nem dos índices: `rotear` é função pura sobre string.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.pipeline import rotear  # noqa: E402


# ----------------------------------------------------------------------
# Rota: contagem x série temporal x evidência
# ----------------------------------------------------------------------
@pytest.mark.parametrize("pergunta", [
    "quais os principais problemas relatados pelos clientes?",
    "do que os clientes mais elogiam?",
    "o que mais reclamam?",
    "quantos clientes reclamaram de atraso?",
    "qual o percentual de reclamações de entrega?",
    "me dê uma visão geral das reclamações",
    "entre as entregas atrasadas, quais os maiores problemas?",
])
def test_pergunta_de_contagem_vai_para_frequencia(pergunta):
    assert rotear(pergunta).tipo == "frequencia"


@pytest.mark.parametrize("pergunta", [
    "a reclamação de não recebimento piorou em algum mês?",
    "a entrega melhorou ao longo do tempo?",
    "qual a evolução das reclamações de atraso?",
    "o atraso aumentou por mês?",
    "qual mês teve mais problema de entrega?",
])
def test_pergunta_temporal_vai_para_serie(pergunta):
    assert rotear(pergunta).tipo == "temporal"


@pytest.mark.parametrize("pergunta", [
    "o que dizem os clientes sobre a caixa amassada?",
    "quem não recebeu o produto, o que relatou?",
    "me mostre reviews sobre produto falsificado",
    "clientes que receberam a cor errada",
])
def test_pergunta_de_evidencia_vai_para_rag(pergunta):
    assert rotear(pergunta).tipo == "evidencia"


def test_pergunta_fora_do_corpus_vai_para_rag():
    """Decisão de desenho: na dúvida, RAG — ele tem o portão atrás.

    Errar para o lado que sabe abster é melhor que errar para o lado que
    afirma número.
    """
    assert rotear("qual a política de troca da Olist?").tipo == "evidencia"
    assert rotear("quem é o presidente do Brasil?").tipo == "evidencia"


def test_temporal_tem_prioridade_sobre_frequencia():
    """"piorou" + "principais" é temporal: a série responde, a contagem não."""
    r = rotear("os principais problemas pioraram em algum mês?")
    assert r.tipo == "temporal"


# ----------------------------------------------------------------------
# Filtros extraídos da pergunta
# ----------------------------------------------------------------------
def test_extrai_uf():
    assert rotear("quais os principais problemas em SP?").estado == "SP"
    assert rotear("reclamações no RJ e principais causas").estado == "RJ"


def test_nao_confunde_palavra_comum_com_uf():
    """UF só casa em maiúsculas e como palavra inteira."""
    assert rotear("quais os principais problemas de sp?").estado is None
    assert rotear("quais os principais problemas de entrega?").estado is None


def test_extrai_filtro_de_atraso():
    assert rotear("entre as entregas atrasadas, quais os principais problemas?").so_atrasadas
    assert rotear("principais problemas de quem recebeu fora do prazo").so_atrasadas
    assert not rotear("quais os principais problemas?").so_atrasadas


def test_distingue_elogio_de_problema():
    assert rotear("do que os clientes mais elogiam?").elogios
    assert not rotear("quais os principais problemas?").elogios


# ----------------------------------------------------------------------
# Identificação do macro-tema
# ----------------------------------------------------------------------
@pytest.mark.parametrize("pergunta,macro", [
    ("a reclamação de não recebimento piorou?", "nao_recebido"),
    ("o atraso de entrega aumentou?", "entrega_atrasada"),
    ("a reclamação de produto quebrado evoluiu?", "produto_danificado"),
    ("as reclamações de cancelamento aumentaram?", "cancelamento_reembolso"),
])
def test_identifica_macro_tema(pergunta, macro):
    assert rotear(pergunta).macro == macro


def test_temporal_sem_tema_explicito_tem_default():
    """Série precisa de UM tema para traçar; sem pista, assume o maior."""
    r = rotear("as reclamações pioraram ao longo do tempo?")
    assert r.tipo == "temporal"
    assert r.macro == "nao_recebido"
