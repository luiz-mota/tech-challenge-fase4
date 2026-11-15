"""Detecção e mascaramento de PII, aplicados na ingestão.

Regra 9 do projeto: dado pessoal não entra no corpus. O corpus alimenta o
prompt do LLM e a tela do `:fontes`, então PII não mascarada aqui vaza nos
dois lugares.

De onde vem
-----------
Os padrões e os dois validadores aritméticos são da aula 06 de IA Generativa
(`aula06-etica-seguranca/handson`). O que ela ensina e que importa aqui: a
regex **propõe** o candidato e a aritmética **decide**. Dígito verificador de
CPF e algoritmo de Luhn para cartão derrubam o falso positivo que a regex
sozinha aceitaria.

O que foi acrescentado, e por quê
---------------------------------
Medição no corpus de 39.388 reviews (14/11/2026), com os padrões da aula sem
alteração:

    documentos com PII validada ......  10  (0,025%)
      PII real ......................   2
      falso positivo ................   8

Os oito falsos positivos têm todos a mesma causa: em review de e-commerce,
oito dígitos seguidos quase nunca são telefone ou CEP. São **número de
pedido** ("Pedido 02-671663252", "Pedido: 01-68698482") e **data** sem
separador ("01092018"). O padrão de telefone da aula casa `\\d{4}[\\s.-]?\\d{4}`
e o de CEP casa `\\d{5}-?\\d{3}`: os dois têm exatamente a forma de um código
de pedido.

Daí a guarda `_RE_PEDIDO`: dígitos precedidos de marcador de pedido não são
tratados como PII. Isso não é conveniência — **número de pedido não é dado
pessoal**, é referência de transação, e é evidência útil numa reclamação de
"pedido não entregue". Mascará-lo destruiria informação legítima sem proteger
ninguém.

A decisão de mascarar mesmo assim
---------------------------------
Sobram falsos positivos (data, erro de digitação) e eles não foram caçados um
a um, porque a assimetria resolve: um falso positivo custa alguns asteriscos
em 0,02% do corpus; um falso negativo publica o telefone de uma pessoa real.
Com esse desequilíbrio, o detector deve errar para o lado de mascarar.

Limitação declarada
-------------------
**Nome de pessoa não é detectado.** Não existe regex para nome, e NER em
português seria um quarto modelo num processo que já está com 0,2 GB de RAM
livre. O que protege parcialmente: o dataset Olist já vem anonimizado pela
origem — nenhum CPF e nenhum e-mail validado nas 39.388 reviews, o que é
consistente com anonimização prévia, não com sorte.
"""

from __future__ import annotations

import re

#: Os cinco padrões da aula 06, sem alteração.
PADROES_PII: dict[str, re.Pattern[str]] = {
    "cpf": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "telefone": re.compile(
        r"(?:\+55\s?)?(?:\(?0?[1-9]\d\)?[\s.-]?)?9?\d{4}[\s.-]?\d{4}\b"),
    "cep": re.compile(r"\b\d{5}-?\d{3}\b"),
    "cartao": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}

#: Marcador de número de pedido. Acrescentado após a medição: referência de
#: transação não é dado pessoal, e é evidência útil numa reclamação.
_RE_PEDIDO = re.compile(
    r"(?:pedido|ordem|protocolo|c[oó]digo|rastreio|nota\s*fiscal|nf)"
    r"\s*(?:n[ºo°]?\.?)?\s*:?\s*[\d.\-/]{4,}", re.IGNORECASE)

#: Tipos verificados primeiro: quem tem validador aritmético reivindica o
#: trecho antes de quem não tem, para não rotular um CPF como telefone.
_ORDEM = ("cartao", "cpf", "email", "cep", "telefone")


def validar_cpf(cpf: str) -> bool:
    """Confere os dois dígitos verificadores. Aritmética, não regex."""
    d = re.sub(r"\D", "", cpf)
    if len(d) != 11 or d == d[0] * 11:
        return False
    for n_dig in (9, 10):
        soma = sum(int(d[i]) * (n_dig + 1 - i) for i in range(n_dig))
        if (soma * 10 % 11) % 10 != int(d[n_dig]):
            return False
    return True


def validar_luhn(numero: str) -> bool:
    """Algoritmo de Luhn, para número de cartão."""
    digitos = [int(x) for x in re.sub(r"\D", "", numero)][::-1]
    if len(digitos) < 13:
        return False
    soma = 0
    for i, d in enumerate(digitos):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        soma += d
    return soma % 10 == 0


VALIDADORES = {"cpf": validar_cpf, "cartao": validar_luhn}


def _faixas_de_pedido(texto: str) -> set[int]:
    """Posições de caractere cobertas por um número de pedido declarado."""
    ocupadas: set[int] = set()
    for m in _RE_PEDIDO.finditer(texto):
        ocupadas |= set(range(m.start(), m.end()))
    return ocupadas


def detectar_pii(texto: str) -> list[dict]:
    """Candidatos a PII, com tipo, trecho, posição e veredito da validação.

    Devolve também os candidatos reprovados (`valido=False`), porque a
    auditoria precisa saber que algo foi **considerado e recusado** — um log
    que só mostra o que passou não permite checar o detector.
    """
    ocorrencias: list[dict] = []
    ocupado = _faixas_de_pedido(texto)   # já começa protegendo o nº de pedido

    for tipo in _ORDEM:
        for m in PADROES_PII[tipo].finditer(texto):
            posicoes = set(range(m.start(), m.end()))
            if posicoes & ocupado:
                continue
            validador = VALIDADORES.get(tipo)
            valido = validador(m.group()) if validador else True
            if valido:
                ocupado |= posicoes
            ocorrencias.append({"tipo": tipo, "trecho": m.group(),
                                "inicio": m.start(), "fim": m.end(),
                                "valido": valido})
    return sorted(ocorrencias, key=lambda o: o["inicio"])


def mascarar(texto: str, ocorrencias: list[dict] | None = None) -> str:
    """Substitui cada PII validada, preservando os dois últimos caracteres.

    Os dois últimos ficam de propósito: quem escreveu reconhece o próprio
    dado e o avaliador vê que houve mascaramento, sem que o dado seja
    recuperável. Substituir por `[REMOVIDO]` apagaria essa evidência.

    A substituição é de trás para frente para que a posição de cada ocorrência
    continue válida enquanto o texto encurta.
    """
    if ocorrencias is None:
        ocorrencias = detectar_pii(texto)
    resultado = texto
    for oc in sorted(ocorrencias, key=lambda o: o["inicio"], reverse=True):
        if not oc["valido"]:
            continue
        trecho = oc["trecho"]
        visivel = trecho[-2:] if len(trecho) > 4 else ""
        mascara = "*" * (len(trecho) - len(visivel)) + visivel
        resultado = resultado[:oc["inicio"]] + mascara + resultado[oc["fim"]:]
    return resultado


def mascarar_serie(textos) -> tuple[list[str], int]:
    """Mascara uma coluna inteira. Devolve os textos e quantos foram alterados."""
    saida, alterados = [], 0
    for t in textos:
        novo = mascarar(t)
        if novo != t:
            alterados += 1
        saida.append(novo)
    return saida, alterados


def tem_pii(texto: str) -> list[str]:
    """Tipos de PII validada presentes. Usado no lado da *pergunta*.

    A aula 06 separa duas políticas: bloquear PII na entrada e mascarar na
    saída. Aqui a entrada é a pergunta do gestor — se ele digitar o próprio
    CPF, isso não deve ir para o prompt nem para log nenhum.
    """
    return sorted({o["tipo"] for o in detectar_pii(texto) if o["valido"]})
