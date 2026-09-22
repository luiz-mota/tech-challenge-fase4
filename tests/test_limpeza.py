"""Garante que a normalização de indexação não destrói evidência.

Cada teste corresponde a uma decisão justificada no README. Se alguém "melhorar"
a limpeza aplicando o pipeline da aula 02 ao texto indexado, estes testes falham.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.knowledge_base.limpeza import (  # noqa: E402
    normalizar_para_indexacao,
    pipeline_nlp_analitico,
)


def test_negacao_sobrevive():
    """O caso que motiva a separação dos dois pipelines."""
    a = normalizar_para_indexacao("Não chegou o produto")
    b = normalizar_para_indexacao("Chegou o produto")
    assert a != b
    assert "Não" in a


def test_pipeline_analitico_perde_a_negacao():
    """Documenta por que o pipeline da aula 02 não pode indexar.

    Não é um defeito: para BoW/TF-IDF a remoção é correta. O teste existe para
    fixar o comportamento e impedir que os dois usos sejam confundidos.
    """
    (a,), (b,) = (
        pipeline_nlp_analitico(["Não chegou o produto"]),
        pipeline_nlp_analitico(["Chegou o produto"]),
    )
    assert a == b  # vetores idênticos para sentidos opostos


def test_digitos_preservados():
    texto = "Comprei dia 25 de fevereiro e hoje dia 29 de março não foi entregue"
    assert "25" in normalizar_para_indexacao(texto)
    assert "29" in normalizar_para_indexacao(texto)


def test_acentos_preservados():
    assert "coração" in normalizar_para_indexacao("Produto de coração partido")


def test_caixa_preservada():
    assert normalizar_para_indexacao("PÉSSIMO ATENDIMENTO") == "PÉSSIMO ATENDIMENTO"


def test_emoji_preservado():
    assert "😍" in normalizar_para_indexacao("Adorei o produto 😍")


def test_ruido_de_superficie_removido():
    sujo = "<p>Adorei!!!!  https://exemplo.com  vc precisa tbm</p>"
    assert normalizar_para_indexacao(sujo) == "Adorei! você precisa também"


def test_contracao_com_pontuacao_colada():
    assert normalizar_para_indexacao("gostei, mas pra mim n serviu.") == (
        "gostei, mas para mim não serviu."
    )


def test_entrada_nao_textual():
    assert normalizar_para_indexacao(None) == ""
    assert normalizar_para_indexacao(float("nan")) == ""
