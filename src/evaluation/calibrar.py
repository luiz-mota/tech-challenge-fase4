"""Calibração do portão de suficiência e comparação das estratégias de busca.

Método da aula 02 de IA Generativa: gabarito de perguntas com documento
correto conhecido, `recall@k`, e comparação entre *vetorial pura*, *BM25 pura*
e *híbrida*. Duas adaptações foram necessárias para o corpus do Olist:

  1. A base da aula tem 5 documentos e o gabarito é escrito à mão. Aqui são
     39.388, então o gabarito usa os **temas** como verdade: a pergunta
     "o que falam sobre atraso?" tem acerto se o top-k contiver pelo menos um
     documento do tema de atraso. É verdade derivada de dado, não de opinião.

  2. A aula não tem portão de recuperação — a recusa dela vive no prompt do
     LLM. Aqui o portão existe antes da geração, e este módulo é o que escolhe
     o número dele.

O achado que motivou o módulo (medido em 14/11): o cosseno máximo NÃO separa
pergunta respondível de pergunta fora do corpus. "quem é o presidente do
Brasil?" casou com "Toalha do maior campeão do Brasil" a 0,752, acima da
pergunta legítima sobre Correios, que ficou em 0,630 — uma palavra em comum
basta para este modelo. Qualquer corte que barre a primeira derruba a segunda.

Por isso o módulo mede vários sinais candidatos e reporta qual separa melhor,
em vez de assumir o cosseno.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.busca import BuscaHibrida, CANDIDATOS_POR_PERNA  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"
REPORTS = RAIZ / "reports"

#: Gabarito. Cada pergunta aponta o tema que DEVE aparecer no top-k. Os temas
#: vêm da camada de agregação e foram validados contra os metadados (os de
#: entrega têm 24–53% de entrega atrasada, contra 10,4% do corpus).
PERGUNTAS_RESPONDIVEIS: list[tuple[str, str]] = [
    ("o que os clientes dizem sobre demora na entrega?", "entrega_atrasada"),
    ("reclamações de quem ainda não recebeu o produto", "nao_recebido"),
    ("clientes que receberam menos unidades do que compraram", "quantidade_errada"),
    ("produto chegou com a caixa amassada", "produto_danificado"),
    ("problemas com os Correios e retirada na agência", "frete_e_retirada"),
    ("tentei entrar em contato e nunca responderam", "atendimento_sem_resposta"),
    ("pedi o cancelamento da compra", "cancelamento_reembolso"),
    ("veio um produto diferente do que eu pedi", "produto_errado"),
    ("entrega foi muito rápida", "elogio_entrega"),
    ("produto entregue dentro do prazo combinado", "elogio_entrega"),
    ("elogios à qualidade do produto", "elogio_produto"),
    ("ainda não testei o produto que recebi", "sem_avaliacao_ainda"),
]

#: Perguntas que o corpus NÃO responde. Três famílias, de propósito:
#: (a) assunto de e-commerce que não existe em review de produto;
#: (b) assunto totalmente alheio;
#: (c) armadilha lexical — compartilha palavra com o corpus sem compartilhar
#:     assunto, que é exatamente o caso que furou o portão.
PERGUNTAS_FORA: list[tuple[str, str]] = [
    ("qual a política de troca e devolução da Olist?", "e-commerce ausente"),
    ("quais as formas de pagamento aceitas no site?", "e-commerce ausente"),
    ("qual o telefone do serviço de atendimento ao cliente?", "e-commerce ausente"),
    ("como faço para me tornar um vendedor na plataforma?", "e-commerce ausente"),
    ("quem é o presidente do Brasil?", "armadilha lexical"),
    ("qual a receita de bolo de chocolate?", "alheio"),
    ("como configurar um servidor de banco de dados?", "alheio"),
    ("qual time ganhou o campeonato brasileiro?", "armadilha lexical"),
    ("qual a previsão do tempo para amanhã em São Paulo?", "armadilha lexical"),
    ("quantos habitantes tem o estado de São Paulo?", "armadilha lexical"),
]

# ----------------------------------------------------------------------
# CONJUNTO DE VALIDAÇÃO — perguntas que NÃO participaram da escolha do corte.
#
# Por que existe: o limiar foi escolhido maximizando acurácia nas 22 perguntas
# acima. Medir nelas mesmas é medir em cima do próprio ajuste, e infla o
# resultado. Estas 16 são disjuntas e nunca entram na escolha do corte — só
# na verificação dele. A diferença entre as duas acurácias é a estimativa
# honesta de quanto o número otimiza para o conjunto em que foi escolhido.
# ----------------------------------------------------------------------
VALIDACAO_RESPONDIVEIS: list[tuple[str, str]] = [
    ("produto nunca chegou no endereço", "nao_recebido"),
    ("passou do prazo e não entregaram", "entrega_atrasada"),
    ("faltou um item no meu pedido", "quantidade_errada"),
    ("embalagem veio toda quebrada", "produto_danificado"),
    ("o material é frágil e de má qualidade", "qualidade_abaixo"),
    ("não é igual à foto do anúncio", "divergente_do_anuncio"),
    ("quero meu dinheiro de volta", "cancelamento_reembolso"),
    ("chegou antes do que eu esperava", "elogio_entrega"),
    ("recomendo, produto excelente", "elogio_produto"),
    ("tive que pagar frete e buscar nos Correios", "frete_e_retirada"),
]

VALIDACAO_FORA: list[tuple[str, str]] = [
    ("qual o prazo de garantia legal no Brasil?", "e-commerce ausente"),
    ("como emito uma segunda via de boleto?", "e-commerce ausente"),
    ("quem fundou a empresa Olist e em que ano?", "e-commerce ausente"),
    ("qual a capital da Austrália?", "alheio"),
    ("como treinar uma rede neural convolucional?", "alheio"),
    ("qual o melhor restaurante de São Paulo?", "armadilha lexical"),
]

K_AVALIACAO = 5


def _sinais(busca: BuscaHibrida, pergunta: str, k: int) -> dict:
    """Os candidatos a sinal do portão, medidos na mesma consulta."""
    r = busca.buscar(pergunta, k=k)
    sim = r.documentos.similaridade.to_numpy() if len(r.documentos) else np.array([0.0])
    return {
        "pergunta": pergunta,
        "cos_max": float(sim.max()),
        "cos_media_topk": float(sim.mean()),
        "cos_mediana_topk": float(np.median(sim)),
        # Concordância entre duas estratégias independentes. A hipótese é que
        # ela separa melhor que o cosseno: BM25 e vetorial não conversam, então
        # concordarem é evidência de que há de fato o que recuperar.
        "sobreposicao": r.diagnostico["sobreposicao"],
        "n_acima_045": int((sim >= 0.45).sum()),
        "n_acima_060": int((sim >= 0.60).sum()),
        "temas_top": r.documentos.get("tema", pd.Series(dtype=float)).tolist(),
        "_docs": r.documentos,
    }


def medir(busca: BuscaHibrida, mapa_tema: dict,
          lista_ok: list[tuple[str, str]], lista_fora: list[tuple[str, str]],
          titulo: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mede os sinais de um par de listas (respondíveis, fora do corpus).

    Recebe as listas por parâmetro em vez de usar as constantes do módulo
    porque roda duas vezes: no conjunto de calibração, que escolhe o corte, e
    no de validação, que nunca participa da escolha.
    """
    print(f"\n[1] Perguntas respondíveis {titulo}")
    linhas_ok = []
    for pergunta, macro_esperado in lista_ok:
        s = _sinais(busca, pergunta, K_AVALIACAO)
        temas = [mapa_tema.get(rid) for rid in s["_docs"].review_id]
        acerto = macro_esperado in temas
        linhas_ok.append({**{k: v for k, v in s.items() if not k.startswith("_")},
                          "macro_esperado": macro_esperado,
                          "macros_recuperados": [t for t in temas if t is not None],
                          "acerto": acerto})
        print(f"  [{'ok   ' if acerto else 'ERROU'}] {macro_esperado:<24} | "
              f"cos_max {s['cos_max']:.3f} | sobrep {s['sobreposicao']:>2} | "
              f"{pergunta[:52]}")

    print(f"\n[2] Perguntas fora do corpus {titulo}")
    linhas_fora = []
    for pergunta, familia in lista_fora:
        s = _sinais(busca, pergunta, K_AVALIACAO)
        linhas_fora.append({**{k: v for k, v in s.items() if not k.startswith("_")},
                            "familia": familia})
        print(f"  cos_max {s['cos_max']:.3f} | sobrep {s['sobreposicao']:>2} | "
              f"{familia:<20} | {pergunta[:44]}")

    return pd.DataFrame(linhas_ok), pd.DataFrame(linhas_fora)


def comparar_sinais(ok: pd.DataFrame, fora: pd.DataFrame) -> pd.DataFrame:
    """Qual sinal separa respondível de não-respondível, e com que folga.

    Critério: acurácia do melhor corte possível naquele sinal. Um sinal cujo
    melhor corte erra é um sinal que não serve, por mais intuitivo que pareça.
    """
    print("\n[3] Qual sinal separa")
    candidatos = ["cos_max", "cos_media_topk", "cos_mediana_topk",
                  "sobreposicao", "n_acima_060"]
    linhas = []
    for sinal in candidatos:
        a, b = ok[sinal].to_numpy(float), fora[sinal].to_numpy(float)
        cortes = np.unique(np.concatenate([a, b]))
        melhor = max(
            ((c, (a >= c).mean() * 0.5 + (b < c).mean() * 0.5) for c in cortes),
            key=lambda par: par[1],
        )
        corte, acuracia = melhor
        linhas.append({
            "sinal": sinal,
            "respondivel_min": a.min(), "respondivel_mediana": np.median(a),
            "fora_max": b.max(), "fora_mediana": np.median(b),
            "corte_otimo": corte, "acuracia": acuracia,
            # Folga: distância entre o pior caso válido e o melhor caso
            # inválido. Positiva = os dois grupos não se tocam.
            "folga": a.min() - b.max(),
        })
    tabela = pd.DataFrame(linhas).sort_values("acuracia", ascending=False)
    print(f"  {'sinal':<20}{'resp.min':>10}{'fora.max':>10}{'folga':>9}"
          f"{'corte':>9}{'acurácia':>10}")
    print("  " + "-" * 68)
    for _, r in tabela.iterrows():
        print(f"  {r.sinal:<20}{r.respondivel_min:>10.3f}{r.fora_max:>10.3f}"
              f"{r.folga:>9.3f}{r.corte_otimo:>9.3f}{r.acuracia * 100:>9.0f}%")
    return tabela


def comparar_estrategias(busca: BuscaHibrida, mapa_tema: dict) -> pd.DataFrame:
    """recall@k das três estratégias — o número que justifica a busca híbrida."""
    import time

    print("\n[4] recall@k por estratégia")
    estrategias = {
        "vetorial pura": lambda p, n: busca._perna_vetorial(p, n)[0],
        "BM25 pura": lambda p, n: busca._perna_lexical(p, n)[0],
        "híbrida (RRF)": None,
    }
    linhas = []
    for nome, ranking_fn in estrategias.items():
        acertos = {1: 0, 3: 0, 5: 0}
        latencias = []
        for pergunta, macro_esperado in PERGUNTAS_RESPONDIVEIS:
            t0 = time.perf_counter()
            if ranking_fn is None:
                docs = busca.buscar(pergunta, k=5).documentos
                ids = docs.review_id.tolist()
            else:
                pos = ranking_fn(pergunta, CANDIDATOS_POR_PERNA)[:5]
                ids = busca.mapa.iloc[pos].review_id.tolist()
            latencias.append((time.perf_counter() - t0) * 1000)
            temas = [mapa_tema.get(i) for i in ids]
            for k in (1, 3, 5):
                acertos[k] += macro_esperado in temas[:k]
        n = len(PERGUNTAS_RESPONDIVEIS)
        linhas.append({"estrategia": nome,
                       "recall@1": acertos[1] / n, "recall@3": acertos[3] / n,
                       "recall@5": acertos[5] / n,
                       "latencia_ms": float(np.mean(latencias))})
        print(f"  {nome:<16} recall@1 {acertos[1]}/{n}  recall@3 {acertos[3]}/{n}"
              f"  recall@5 {acertos[5]}/{n}  {np.mean(latencias):.0f} ms")
    return pd.DataFrame(linhas)


def aplicar_corte(ok: pd.DataFrame, fora: pd.DataFrame, sinal: str,
                  corte: float, rotulo: str) -> dict:
    """Acurácia de um corte JÁ ESCOLHIDO, aplicado a um conjunto qualquer."""
    acertos_ok = (ok[sinal] >= corte).sum()
    acertos_fora = (fora[sinal] < corte).sum()
    n = len(ok) + len(fora)
    res = {
        "conjunto": rotulo, "sinal": sinal, "corte": corte,
        "respondiveis_aceitas": f"{acertos_ok}/{len(ok)}",
        "fora_recusadas": f"{acertos_fora}/{len(fora)}",
        "acuracia": (acertos_ok + acertos_fora) / n,
    }
    print(f"  {rotulo:<24} aceitou {acertos_ok}/{len(ok)} válidas, "
          f"recusou {acertos_fora}/{len(fora)} inválidas  "
          f"-> {res['acuracia'] * 100:.0f}%")
    return res


def main() -> None:
    mapa = pd.read_parquet(PROCESSED / "corpus_temas.parquet")
    mapa_tema = dict(zip(mapa.review_id, mapa.macro))
    busca = BuscaHibrida()
    print(f"índice: {busca.indice.ntotal:,} vetores | gabarito por MACRO-TEMA")

    print("\n" + "=" * 70)
    print("CALIBRAÇÃO — escolhe o corte")
    print("=" * 70)
    ok, fora = medir(busca, mapa_tema, PERGUNTAS_RESPONDIVEIS, PERGUNTAS_FORA,
                     "(calibração)")
    sinais = comparar_sinais(ok, fora)
    melhor = sinais.iloc[0]

    print("\n" + "=" * 70)
    print("VALIDAÇÃO — perguntas que não participaram da escolha")
    print("=" * 70)
    ok_v, fora_v = medir(busca, mapa_tema, VALIDACAO_RESPONDIVEIS,
                         VALIDACAO_FORA, "(validação)")

    print("\n[4] O corte escolhido, nos dois conjuntos")
    comparacao = pd.DataFrame([
        aplicar_corte(ok, fora, melhor.sinal, melhor.corte_otimo, "calibração"),
        aplicar_corte(ok_v, fora_v, melhor.sinal, melhor.corte_otimo, "validação"),
    ])
    queda = (comparacao.acuracia.iloc[0] - comparacao.acuracia.iloc[1]) * 100
    print(f"  queda da calibração para a validação: {queda:+.0f} p.p.")

    estrategias = comparar_estrategias(busca, mapa_tema)

    REPORTS.mkdir(parents=True, exist_ok=True)
    for nome, df in [("calibracao_respondiveis", ok), ("calibracao_fora", fora),
                     ("validacao_respondiveis", ok_v), ("validacao_fora", fora_v),
                     ("calibracao_sinais", sinais),
                     ("calibracao_generalizacao", comparacao),
                     ("comparacao_estrategias", estrategias)]:
        df.drop(columns=[c for c in ("temas_top",) if c in df.columns]).to_csv(
            REPORTS / f"{nome}.csv", index=False, encoding="utf-8-sig")

    print(f"\n[6] Conclusão")
    print(f"  sinal: {melhor.sinal}, corte {melhor.corte_otimo:.3f}")
    print(f"  acurácia na calibração {comparacao.acuracia.iloc[0] * 100:.0f}%, "
          f"na validação {comparacao.acuracia.iloc[1] * 100:.0f}%")
    print(f"  relatórios em reports/")


if __name__ == "__main__":
    main()
