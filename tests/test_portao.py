"""Testes do portão de suficiência, nos três graus da regra 3.

`_avaliar` é estático e recebe um DataFrame — então dá para testar a decisão
do portão sem carregar o FAISS, o BM25 e o modelo de embedding. Isso é de
propósito: a regra de decisão é a parte que precisa de teste rápido, e ela
não depende de como os documentos foram encontrados.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.retrieval.busca import (  # noqa: E402
    DF_MINIMO_ASSUNTO,
    LIMIAR_SUFICIENCIA,
    BuscaHibrida,
)


def docs(similaridades: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"similaridade": similaridades})


def avaliar(similaridades: list[float], df_assunto: int = 0):
    return BuscaHibrida._avaliar(docs(similaridades), [], df_assunto)


# ------------------------------------------------- caso 1: nenhuma evidência
def test_nenhuma_via_passa_entao_nao_chama_o_llm():
    """Pergunta fora do corpus: cosseno baixo e termo central raro.

    Este é o caso que protege contra a pergunta "quem é o presidente do
    Brasil?" — a palavra "Brasil" existe em algumas reviews, mas em 34
    documentos, longe dos 150 que marcam assunto do corpus.
    """
    suficiente, grau, motivo = avaliar([0.62, 0.60, 0.58, 0.55, 0.50], df_assunto=34)
    assert suficiente is False
    assert grau == "nenhuma"
    assert "fora do corpus" in motivo


def test_documentos_vazios():
    suficiente, grau, _ = BuscaHibrida._avaliar(
        pd.DataFrame({"similaridade": []}), [], 0)
    assert suficiente is False
    assert grau == "nenhuma"


def test_uma_evidencia_so_e_anedota_nao_evidencia():
    """Passou pelo vocabulário mas só um documento é citável.

    Uma review parecida pode ser coincidência. A resposta cita evidência, não
    anedota — e por isso o filtro vale para as duas vias do portão.
    """
    suficiente, grau, motivo = avaliar([0.80, 0.40, 0.35, 0.30, 0.20], df_assunto=900)
    assert suficiente is False
    assert grau == "nenhuma"
    assert "anedota" in motivo


# -------------------------------------------------- caso 2: evidência fraca
def test_passar_so_pelo_vocabulario_e_evidencia_fraca():
    """O corpus fala do assunto, mas nada se parece muito com a pergunta.

    É o caso de "reclamações sobre os Correios": mediana 0,553, e o termo
    central em 1.031 documentos. A resposta é dada — calar seria errado,
    porque a evidência existe — mas com ressalva, porque passar raspando e
    passar com folga não são a mesma situação.
    """
    suficiente, grau, motivo = avaliar(
        [0.60, 0.58, 0.55, 0.52, 0.51], df_assunto=1031)
    assert suficiente is True
    assert grau == "baixa"
    assert "vocabulário" in motivo
    assert "FRACA" in motivo


def test_poucos_citaveis_e_evidencia_fraca_mesmo_com_semantica_alta():
    """Mediana alta não basta se a maioria do que foi lido não se sustenta.

    Três citáveis em cinco quer dizer que dois dos documentos que o modelo vai
    ler estão abaixo do limiar de documento. A afirmação fica sustentada pela
    minoria da evidência.
    """
    suficiente, grau, motivo = avaliar(
        [0.95, 0.90, 0.75, 0.40, 0.30], df_assunto=900)
    assert suficiente is True
    assert grau == "baixa"
    assert "3 de 5" in motivo


# -------------------------------------------------- caso 3: evidência forte
def test_semantica_alta_e_quatro_citaveis_e_evidencia_forte():
    suficiente, grau, motivo = avaliar(
        [0.95, 0.90, 0.85, 0.80, 0.40], df_assunto=900)
    assert suficiente is True
    assert grau == "alta"
    assert "FRACA" not in motivo


def test_as_duas_vias_juntas_sao_registradas():
    """A resposta tem de dizer por qual via passou — é o que a torna auditável."""
    _, _, motivo = avaliar([0.95, 0.92, 0.90, 0.88, 0.85], df_assunto=900)
    assert "semântica e vocabulário" in motivo


# ------------------------------------------------------ fronteiras exatas
def test_limiar_semantico_e_inclusivo():
    """`>=` e não `>`: o limiar foi calibrado como valor que passa.

    Uma pergunta respondível do conjunto de calibração ficou exatamente em
    0,689, e tratá-la como reprovada mudaria o resultado da calibração.
    """
    _, grau, _ = avaliar([LIMIAR_SUFICIENCIA] * 5, df_assunto=0)
    assert grau == "alta"


def test_corte_de_vocabulario_e_inclusivo():
    suficiente, _, _ = avaliar([0.60] * 5, df_assunto=DF_MINIMO_ASSUNTO)
    assert suficiente is True
    suficiente, _, _ = avaliar([0.60] * 5, df_assunto=DF_MINIMO_ASSUNTO - 1)
    assert suficiente is False


@pytest.mark.parametrize("grau", ["alta", "baixa", "nenhuma"])
def test_os_tres_graus_existem_e_sao_os_unicos(grau):
    """A regra 3 pede três casos distintos, não dois. Fixa o vocabulário."""
    resultados = {
        avaliar([0.95, 0.90, 0.85, 0.80, 0.40], 900)[1],
        avaliar([0.60, 0.58, 0.55, 0.52, 0.51], 1031)[1],
        avaliar([0.62, 0.60, 0.58, 0.55, 0.50], 34)[1],
    }
    assert resultados == {"alta", "baixa", "nenhuma"}
    assert grau in resultados
