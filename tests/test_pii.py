"""Testes do mascaramento de PII.

Cada teste fixa uma decisão medida no corpus de 39.388 reviews, não um
comportamento genérico de regex. Os casos vêm de duas fontes: as frases
canônicas da aula 06 de IA Generativa e os 10 documentos que a medição de
14/11/2026 encontrou no corpus real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.knowledge_base.pii import (  # noqa: E402
    detectar_pii,
    mascarar,
    mascarar_serie,
    tem_pii,
    validar_cpf,
    validar_luhn,
)


# ---------------------------------------------------------------- validação
def test_cpf_com_digito_verificador_errado_e_recusado():
    """A regex propõe, a aritmética decide.

    `123.456.789-00` tem a forma de um CPF e não é um CPF. Mascarar isso seria
    apagar texto legítimo por semelhança de formato — é exatamente o erro que
    o validador existe para evitar.
    """
    assert validar_cpf("529.982.247-25")
    assert not validar_cpf("123.456.789-00")
    assert not validar_cpf("111.111.111-11")   # todos iguais

    ocs = detectar_pii("Anotei o CPF 123.456.789-00 do formulario.")
    assert [o["valido"] for o in ocs] == [False]
    assert mascarar("Anotei o CPF 123.456.789-00 do formulario.") == (
        "Anotei o CPF 123.456.789-00 do formulario.")


def test_luhn_recusa_numero_de_cartao_invalido():
    assert validar_luhn("4539 1488 0343 6467")
    assert not validar_luhn("4539 1488 0343 6468")


# ------------------------------------------------- a guarda de nº de pedido
@pytest.mark.parametrize("texto", [
    "Pedido 02-671663252. Cancelei o pedido e o mesmo nao foi estornado.",
    "Pedido: 01-68698482 de 21",
    "codigo de rastreio 12345678 nao funciona",
    "protocolo 98765432 aberto e sem resposta",
])
def test_numero_de_pedido_nao_e_mascarado(texto):
    """Número de pedido não é dado pessoal — é referência de transação.

    Esta é a diferença entre o detector da aula e o daqui, e ela veio de
    medição: dos 10 documentos que o detector da aula marcou no corpus, 8 eram
    falso positivo e a causa era sempre a mesma — em review de e-commerce,
    oito dígitos seguidos são código de pedido, não telefone nem CEP.

    Mascarar destruiria evidência útil ("pedido 02-671663252 não chegou") sem
    proteger pessoa nenhuma.
    """
    assert mascarar(texto) == texto


# --------------------------------------------------- PII real é mascarada
def test_telefone_em_review_real_e_mascarado():
    """O caso que justifica a regra existir.

    Esta frase está no corpus. Sem mascaramento, o número vai para o prompt do
    LLM e para a tela do `:fontes`.
    """
    texto = "Faltou item descrito Me ligue 34 99978 4951"
    assert mascarar(texto) != texto
    assert "99978" not in mascarar(texto)


def test_email_e_cpf_validos_sao_mascarados():
    texto = "Meu CPF e 529.982.247-25, fale em ana.souza@empresa.com.br"
    saida = mascarar(texto)
    assert "529.982.247" not in saida
    assert "ana.souza@empresa" not in saida


def test_mascara_preserva_os_dois_ultimos_caracteres():
    """Decisão de formato: `****25`, não `[REMOVIDO]`.

    Os dois últimos deixam o autor reconhecer o próprio dado e deixam o
    avaliador ver que houve mascaramento. `[REMOVIDO]` apagaria a evidência de
    que a proteção agiu, e o tamanho do trecho.
    """
    saida = mascarar("Meu CPF e 529.982.247-25 ok")
    assert "25 ok" in saida
    assert saida.count("*") == len("529.982.247-25") - 2


# ------------------------------------------------- review normal fica intacta
@pytest.mark.parametrize("texto", [
    "produto chegou com a caixa amassada e vou dar de presente",
    "comprei 2 unidades, paguei 150,00 e nota 10 para o vendedor",
    "entrega em 3 dias, recomendo",
    "a tv de 50 polegadas chegou com a tela trincada",
])
def test_review_sem_pii_nao_e_tocada(texto):
    """Falso positivo em review comum seria caro: são 39.388 documentos."""
    assert mascarar(texto) == texto


def test_mascarar_serie_conta_quantos_mudaram():
    textos = ["caixa amassada", "Me ligue 34 99978 4951", "pedido 02-671663252"]
    saida, alterados = mascarar_serie(textos)
    assert alterados == 1
    assert saida[0] == textos[0]
    assert saida[2] == textos[2]


# ------------------------------------------------------- lado da pergunta
def test_pergunta_com_cpf_e_sinalizada():
    """A aula 06 bloqueia PII na entrada; aqui a entrada é a pergunta.

    Um CPF na pergunta não é acidente de formatação: é a intenção de consultar
    uma pessoa. Este sistema responde sobre padrões, não sobre indivíduos.
    """
    assert tem_pii("meu CPF e 529.982.247-25, pode consultar?") == ["cpf"]


def test_pergunta_legitima_com_numero_de_pedido_passa():
    assert tem_pii("quantos clientes reclamaram do pedido 02-671663252?") == []


def test_pergunta_normal_passa():
    assert tem_pii("quais os principais problemas relatados?") == []
