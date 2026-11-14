"""Testes da CLI.

A CLI não é testada ponta a ponta aqui de propósito: isso carregaria o índice
FAISS e o modelo de geração, e um teste que leva 18 s para subir não é rodado.
O que se testa é o que quebra sem avisar — o parser de argumentos, a lista de
comandos e a recusa antecipada de uma combinação impossível.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import app  # noqa: E402


def test_ajuda_cobre_as_tres_rotas():
    """Quem lê `:ajuda` tem de descobrir que existem três caminhos.

    O sistema não é um chat único: pergunta de contagem não passa pelo LLM.
    Se a ajuda esconde isso, o avaliador supõe que tudo é geração.
    """
    for rota in ("contagem", "temporal", "citação"):
        assert rota in app.AJUDA


def test_ajuda_lista_todos_os_comandos_implementados():
    """Comando implementado e não documentado é comando que ninguém usa."""
    for cmd in (":ajuda", ":temas", ":demo", ":fontes", ":rota", ":sair"):
        assert cmd in app.AJUDA


def test_sem_llm_recusa_pergunta_de_evidencia_antes_de_carregar():
    """Falhar rápido: a recusa vem do roteador, não de um crash 18 s depois.

    `--sem-llm` existe para rodar em máquina apertada. Se a pergunta exige o
    modelo, o certo é dizer isso em milissegundos — carregar o corpus inteiro
    para então levantar `RuntimeError` é gastar o tempo de quem rodou.
    """
    r = subprocess.run(
        [sys.executable, "-X", "utf8", str(RAIZ / "app.py"), "--sem-llm",
         "-p", "o que dizem os clientes sobre a caixa amassada?"],
        capture_output=True, text=True, timeout=60, cwd=RAIZ,
    )
    assert r.returncode == 2
    assert "rota de evidência" in r.stdout
    # não chegou a carregar nada
    assert "carregando" not in r.stdout


def test_parser_aceita_as_quatro_formas_de_uso():
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()) as saida:
        with pytest.raises(SystemExit):
            sys.argv = ["app.py", "--help"]
            app.main()
    texto = saida.getvalue()
    for flag in ("--pergunta", "--demo", "--sem-llm", "--fontes"):
        assert flag in texto
