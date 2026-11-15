"""Interface de linha de comando do Voice of Customer Intelligence.

Por que CLI e não Streamlit
---------------------------
A máquina de desenvolvimento tem 7,9 GB de RAM com 0,2 GB livres (medido em
14/11), e o processo já carrega quatro coisas grandes: o índice FAISS com
39.388 vetores, o BM25 serializado, o corpus em pandas e dois modelos
(MiniLM para embedding, Qwen 0.5B para geração). Streamlit acrescenta um
servidor e um segundo processo Python ao mesmo orçamento — a mesma causa que
tornou a geração 50x mais lenta na medição do dia 14.

A CLI também tem uma vantagem de avaliação: ela mostra a *decisão*, não só a
resposta. Cada pergunta imprime qual rota foi escolhida, por qual via o portão
de suficiência passou, e o texto integral de cada review citada — o que é
exatamente o que precisa ser auditado. Uma caixa de chat esconde isso.

Uso
---
    python app.py                      # modo interativo
    python app.py -p "sua pergunta"    # uma pergunta e sai
    python app.py --demo               # roda o roteiro de demonstração
    python app.py --sem-llm            # só roteamento e contagem (rápido)

No modo interativo, comandos começam com `:` — digite `:ajuda`.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path

# Silenciado antes de qualquer import de `transformers`: as barras de progresso
# de carregamento de peso vão para stderr e sujam a demonstração. Precisa ser
# variável de ambiente, não chamada de função, porque a biblioteca lê isso no
# momento do import.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402

from src.generation.agregacao import RespostaAgregada  # noqa: E402
from src.generation.pipeline import (  # noqa: E402
    PERGUNTAS_DEMO,
    Assistente,
    PerguntaBloqueada,
    rotear,
)
from src.generation.responder import Resposta  # noqa: E402

LARGURA = 78
REGUA = "=" * LARGURA

AJUDA = """\
Comandos
  :ajuda              esta lista
  :temas              os 16 macro-temas e quais contam como problema
  :demo               roda o roteiro de perguntas de demonstração
  :fontes             mostra o texto integral das reviews citadas na última resposta
  :rota <pergunta>    só mostra para onde o roteador manda, sem executar
  :sair               encerra

Qualquer outra linha é tratada como pergunta.

Três tipos de pergunta, três caminhos
  "quais os principais problemas?"        -> contagem (pandas, sem LLM)
  "o não recebimento piorou em algum mês?" -> série temporal (pandas, sem LLM)
  "o que dizem sobre a caixa amassada?"   -> RAG com citação (busca + LLM)
"""


# ----------------------------------------------------------------------
def _cabecalho(sem_llm: bool) -> None:
    print(REGUA)
    print("VOICE OF CUSTOMER INTELLIGENCE — reviews Olist")
    print("Tech Challenge Fase 4 · FIAP Pós-Tech AI Scientist")
    print(REGUA)
    if sem_llm:
        print("modo --sem-llm: rotas de contagem e temporal apenas.")
        print("Perguntas de evidência vão avisar que o LLM não está carregado.")
    print()


def _mostrar_temas() -> None:
    from src.generation.agregacao import Agregador

    ag = Agregador()
    print(ag.principais("todos os temas de problema", top=13))
    print()
    print(ag.principais("todos os temas de elogio", elogios=True, top=3))
    print()
    print("  Fora da contagem de problemas: 'misto' (crítica e elogio na mesma")
    print("  frase), 'sem_avaliacao_ainda' (cliente diz que não testou) e")
    print("  'sem_tema' (nenhum centroide passou de 0,55 de similaridade).")
    print("  Eles aparecem no denominador declarado, não no numerador — o")
    print("  sistema conta o que afirma, e não afirma o que não leu.")


def _mostrar_fontes(assistente: Assistente, ids: list[str]) -> None:
    """Texto integral de cada review citada.

    É aqui que a rastreabilidade deixa de ser promessa: o `review_id` que a
    resposta cita é procurado no mapa de citação e o texto original aparece,
    com nota, categoria e se a entrega atrasou. Quem avalia pode conferir
    frase por frase se a resposta se sustenta na evidência.
    """
    if not ids:
        print("  (a última resposta não citou nenhuma review)")
        return
    if assistente.gerador is None:
        print("  (mapa de citação indisponível em --sem-llm)")
        return

    mapa = assistente.gerador.busca.mapa
    linhas = mapa[mapa.review_id.isin(ids)]
    for r in linhas.itertuples():
        atraso = (f"atrasou {r.atraso_dias:.0f} dias"
                  if r.entrega_atrasada else "entregue no prazo")
        # A data entra porque reclamação de entrega é datada: "não recebi" em
        # dezembro de 2017 e em março de 2018 são os dois picos medidos, e sem
        # a data o avaliador não consegue ligar a evidência à série temporal.
        data = pd.to_datetime(r.data_review).strftime("%d/%m/%Y")
        print(f"  [{r.review_id[:8]}] {data} · nota {r.review_score}"
              f" · {r.categoria} · {r.customer_state} · {atraso}")
        for linha in textwrap.wrap(r.texto, LARGURA - 6):
            print(f"      {linha}")
        print()


# ----------------------------------------------------------------------
def responder(assistente: Assistente, pergunta: str) -> list[str]:
    """Executa uma pergunta e imprime a resposta. Devolve os ids citados."""
    t0 = time.perf_counter()
    try:
        rota, resposta = assistente.perguntar(pergunta)
    except PerguntaBloqueada as e:     # política de entrada: recusa antes de buscar
        print(REGUA)
        print(f'PERGUNTA: "{pergunta[:60]}{"..." if len(pergunta) > 60 else ""}"')
        print(f"  BLOQUEADA na entrada: {e}")
        print(f"\n  tempo total: {(time.perf_counter() - t0) * 1000:.0f} ms\n")
        return []
    except RuntimeError as e:          # rota de evidência sem LLM carregado
        print(f"  não posso responder nesta configuração: {e}")
        return []
    ms = (time.perf_counter() - t0) * 1000

    print(REGUA)
    print(f"ROTA: {rota}")
    print("-" * LARGURA)
    print(resposta)

    if isinstance(resposta, Resposta) and not resposta.respondeu:
        # A abstenção é resultado, não falha. Dizer de onde ela veio é o que
        # separa "o sistema não sabe" de "o sistema quebrou".
        de_onde = {
            "portao": "a busca não encontrou base suficiente no corpus",
            "modelo": "a busca passou, mas o modelo não achou a resposta "
                      "dentro das evidências",
            "guardrail": "o modelo respondeu sem citar nenhuma evidência "
                         "válida — resposta sem fonte é rejeitada por código",
        }
        print(f"\n  por que absteve: {de_onde.get(resposta.abstencao_por, '?')}")

    print(f"\n  tempo total: {ms:.0f} ms")
    print()

    if isinstance(resposta, Resposta):
        return resposta.citacoes
    if isinstance(resposta, RespostaAgregada):
        return resposta.citacoes
    return []


def interativo(assistente: Assistente, sem_llm: bool) -> None:
    _cabecalho(sem_llm)
    print(AJUDA)
    ultimas: list[str] = []

    while True:
        try:
            linha = input("pergunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not linha:
            continue

        if linha.startswith(":"):
            cmd, _, arg = linha[1:].partition(" ")
            cmd = cmd.lower()
            if cmd in {"sair", "q", "quit", "exit"}:
                return
            if cmd in {"ajuda", "h", "help"}:
                print(AJUDA)
            elif cmd == "temas":
                _mostrar_temas()
            elif cmd == "fontes":
                _mostrar_fontes(assistente, ultimas)
            elif cmd == "rota":
                print(f"  {rotear(arg)}" if arg.strip()
                      else "  uso: :rota <pergunta>")
            elif cmd == "demo":
                for p in PERGUNTAS_DEMO:
                    print(f"\n>>> {p}")
                    ultimas = responder(assistente, p)
            else:
                print(f"  comando desconhecido: {cmd!r} — tente :ajuda")
            print()
            continue

        ultimas = responder(assistente, linha)
        if ultimas:
            print("  digite :fontes para ler as reviews citadas na íntegra\n")


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Voice of Customer Intelligence — RAG sobre reviews Olist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=AJUDA,
    )
    ap.add_argument("-p", "--pergunta", help="responde uma pergunta e sai")
    ap.add_argument("--demo", action="store_true",
                    help="roda o roteiro de perguntas de demonstração")
    ap.add_argument("--sem-llm", action="store_true", dest="sem_llm",
                    help="não carrega o modelo de geração (rotas de contagem só)")
    ap.add_argument("--fontes", action="store_true",
                    help="com -p, já imprime as reviews citadas na íntegra")
    args = ap.parse_args()

    if args.sem_llm and args.pergunta and rotear(args.pergunta).tipo == "evidencia":
        print("essa pergunta vai para a rota de evidência, que exige o LLM. "
              "Rode sem --sem-llm.")
        return 2

    print("carregando índices e modelos...", flush=True)
    t0 = time.perf_counter()
    assistente = Assistente(carregar_llm=not args.sem_llm)
    print(f"pronto em {time.perf_counter() - t0:.1f} s\n")

    if args.pergunta:
        ids = responder(assistente, args.pergunta)
        if args.fontes:
            _mostrar_fontes(assistente, ids)
        return 0

    if args.demo:
        _cabecalho(args.sem_llm)
        for p in PERGUNTAS_DEMO:
            print(f">>> {p}")
            responder(assistente, p)
        return 0

    interativo(assistente, args.sem_llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
