"""Busca híbrida: BM25 + vetorial fundidos por RRF, com portão de suficiência.

O caminho de uma pergunta:

    pergunta
       |
       +-- BM25 sobre texto_bm25 ....... ranking lexical
       +-- FAISS sobre embeddings ...... ranking semântico
       |
       RRF (k=60) ..................... um ranking só
       |
       portão de suficiência .......... passa, ou abstém
       |
       documentos com review_id

Por que RRF e não soma de escores: BM25 devolve escore sem limite superior
(depende de idf e comprimento) e cosseno devolve algo entre -1 e 1. Somar
exige normalizar, e toda normalização assume uma distribuição que muda a cada
pergunta. O RRF ignora o escore e usa só a POSIÇÃO:

    rrf(doc) = soma_das_pernas( 1 / (k + posicao_na_perna) )

k=60 é o valor do artigo original (Cormack et al., 2009) e amortece a
diferença entre as primeiras posições — o 1º lugar não vale desproporcional-
mente mais que o 3º, o que importa quando uma das pernas está errada.

O portão de suficiência é a mesma regra que governou a realocação de outliers
na camada de temas: melhor admitir que não sei do que afirmar com confiança
falsa. Ele olha a similaridade de cosseno real, não a posição no RRF — posição
sempre existe, mesmo quando nada presta.
"""

from __future__ import annotations

import pickle
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.knowledge_base.limpeza import (  # noqa: E402
    normalizar_para_indexacao,
    pipeline_lexical_bm25,
)

RAIZ = Path(__file__).resolve().parents[2]
INDEX = RAIZ / "data" / "index"
MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

#: Constante do RRF. 60 é o valor do artigo original.
RRF_K = 60

#: Quantos candidatos cada perna devolve antes da fusão. Maior que o `k` final
#: de propósito: a fusão só consegue promover um documento que pelo menos uma
#: das pernas trouxe.
CANDIDATOS_POR_PERNA = 50

#: Calibrado em 14/11/2026 por `src/evaluation/calibrar.py` sobre 22 perguntas
#: (12 respondíveis, 10 fora do corpus): 96% de acerto, o melhor entre cinco
#: sinais testados.
#:
#: O sinal é a MEDIANA do cosseno do top-k, não o máximo. Medição que obrigou
#: a mudança: "quem é o presidente do Brasil?" casou com "Toalha do maior
#: campeão do Brasil" a 0,752 de cosseno máximo — acima da pergunta legítima
#: sobre Correios, que ficou em 0,630. Nenhum corte sobre o máximo separa os
#: dois, porque uma palavra em comum basta para produzir um acerto isolado
#: alto. A mediana exige que o CONJUNTO recuperado seja bom, e coincidência
#: lexical não acontece cinco vezes.
#:
#: Descartada: a sobreposição entre as duas pernas. Era a hipótese inicial e
#: foi o PIOR dos cinco sinais (82%) — "qual o telefone do atendimento?"
#: produziu sobreposição 5 (está fora do corpus) enquanto "quem ainda não
#: recebeu" produziu 0 (está dentro). As pernas concordam até sobre ruído,
#: quando o ruído tem palavras frequentes.
#:
#: Os grupos ainda se tocam (folga -0,083), então o portão erra. Por isso ele
#: é a PRIMEIRA de duas camadas de abstenção: a segunda é a instrução de
#: recusa no prompt de geração (método da aula 02 de IA Generativa).
LIMIAR_SUFICIENCIA = 0.689

#: Piso por documento, usado só para contar quantos sustentam a resposta.
#: Mais baixo que o limiar de propósito: o portão julga o conjunto, esta
#: constante julga se vale citar aquele documento específico.
LIMIAR_DOCUMENTO = 0.50

#: Moldura da pergunta: palavras que descrevem o ato de perguntar, não o
#: assunto perguntado. Sem removê-las, "reclamações sobre os Correios" busca
#: por `reclam` e devolve "Sem reclamações" — documento de 2 palavras, que a
#: normalização de comprimento do BM25 favorece. Removidas apenas da perna
#: LEXICAL; a vetorial recebe a frase inteira, porque ali a sintaxe carrega
#: sentido ("quem não recebeu" ≠ "quem recebeu").
MOLDURA_PERGUNTA = frozenset({
    "reclamação", "reclamações", "reclamar", "cliente", "clientes",
    "dizer", "dizem", "falam", "falar", "comentam", "comentar",
    "avaliação", "avaliações", "review", "reviews", "opinião", "opiniões",
    "quais", "qual", "quantos", "quantas", "quem", "sobre", "respeito",
    "mencionam", "citam", "relatam", "principais", "exemplo", "exemplos",
})

#: A mesma lista sem acento, para a comparação não depender de o usuário
#: digitar "reclamações" ou "reclamacoes". Derivada, não escrita à mão: lista
#: duplicada sai de sincronia na primeira manutenção.
_MOLDURA_SEM_ACENTO = frozenset(
    "".join(c for c in unicodedata.normalize("NFKD", p) if not unicodedata.combining(c))
    for p in MOLDURA_PERGUNTA
)

#: IDF mínimo para um termo da pergunta ser considerado discriminativo. O
#: termo de maior IDF da consulta passa a ser OBRIGATÓRIO no documento: sem
#: isso, documento curto que casa só o termo banal vence por normalização de
#: comprimento. É o que separa "Correios" de "reclamações" na mesma consulta.
IDF_MINIMO_OBRIGATORIO = 2.0

#: Quantos documentos, no mínimo, precisam passar o limiar. Um único documento
#: parecido pode ser coincidência; a resposta cita evidência, não anedota.
MIN_DOCS_SUFICIENTES = 2

#: Segunda via do portão: quantos documentos do corpus precisam conter o termo
#: central da pergunta para que ela conte como "assunto que o corpus trata".
#:
#: Por que uma segunda via existe: a mediana do cosseno acerta 37 das 38
#: perguntas medidas, mas a falha dela não é aleatória — ela derruba pergunta
#: legítima de formulação CURTA, porque pergunta curta recupera documento
#: curto e documento curto tem cosseno menor. "reclamações sobre os Correios"
#: recuperava "Problemas com os Correios" e "NÃO CHEGOU POR PROBLEMAS NOS
#: CORREIOS" — recuperação certa, portão errado.
#:
#: Este sinal é de natureza diferente: não é similaridade, é **cobertura de
#: vocabulário**. Se a palavra central da pergunta quase não existe no corpus,
#: a pergunta é de outro assunto — e nenhuma similaridade conserta isso. É o
#: que a palavra "Brasil" numa review de toalha consegue falsificar no cosseno
#: e não consegue falsificar aqui.
#:
#: Medido nas 38 perguntas: respondíveis com DF mediano de 281 (máximo 2.515),
#: fora do corpus com DF mediano de 12 e **máximo de 105**. O corte em 150
#: deixa 45 de margem sobre o pior caso inválido, e não perde nenhuma pergunta
#: válida que a mediana já não cobrisse.
#:
#: Sozinho este sinal é PIOR que a mediana (92% contra 97%): serve como via
#: alternativa, não como substituto.
DF_MINIMO_ASSUNTO = 150


@dataclass
class Resultado:
    """O que a busca devolve. `suficiente=False` significa abster."""

    pergunta: str
    documentos: pd.DataFrame
    suficiente: bool
    motivo: str
    melhor_similaridade: float
    diagnostico: dict = field(default_factory=dict)

    def __str__(self) -> str:
        cab = (f"[{'OK' if self.suficiente else 'ABSTEM'}] {self.pergunta!r}\n"
               f"  {self.motivo} (melhor cosseno {self.melhor_similaridade:.3f})")
        if self.documentos.empty:
            return cab
        linhas = [
            f"  {r.review_id[:10]} | nota {r.review_score} | cos {r.similaridade:.3f}"
            f" | rrf {r.rrf:.4f} | {r.texto[:72]}"
            for r in self.documentos.itertuples()
        ]
        return cab + "\n" + "\n".join(linhas)


class BuscaHibrida:
    """Carrega os índices uma vez e responde consultas."""

    def __init__(self) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self.indice = faiss.read_index(str(INDEX / "faiss_corpus.index"))
        with open(INDEX / "bm25_corpus.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.mapa = pd.read_parquet(INDEX / "mapa_citacao.parquet")
        self.modelo = SentenceTransformer(MODELO_EMBEDDING)
        self._faiss = faiss
        self._cache_contem: dict[str, np.ndarray] = {}

    def _docs_com(self, termo: str) -> np.ndarray:
        """Máscara booleana dos documentos que contêm o termo.

        Calculada sob demanda e memorizada. Um índice invertido completo
        (9.401 termos × 39.388 docs) seria 370 milhões de booleanos para um
        ganho que não existe: cada consulta precisa de um termo só.
        """
        if termo not in self._cache_contem:
            self._cache_contem[termo] = np.fromiter(
                (termo in d for d in self.bm25.doc_freqs),
                dtype=bool, count=len(self.bm25.doc_freqs),
            )
        return self._cache_contem[termo]

    # ------------------------------------------------------------------
    # As duas pernas
    # ------------------------------------------------------------------
    def _perna_vetorial(self, pergunta: str, n: int):
        """A pergunta passa pela MESMA rota de limpeza que o corpus passou.

        Se o corpus foi indexado com acento e caixa preservados e a pergunta
        for normalizada de outro jeito, consulta e índice deixam de falar a
        mesma língua. Essa assimetria é um erro clássico e silencioso.
        """
        vetor = self.modelo.encode(
            [normalizar_para_indexacao(pergunta)]
        ).astype("float32")
        self._faiss.normalize_L2(vetor)
        sim, pos = self.indice.search(vetor, n)
        return pos[0], sim[0]

    @staticmethod
    def _sem_acento(palavra: str) -> str:
        nfkd = unicodedata.normalize("NFKD", palavra)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def _tokens_consulta(self, pergunta: str) -> list[str]:
        """Tokens da pergunta sem a moldura, no mesmo espaço do índice.

        A moldura é removida ANTES da stemização, nas palavras originais, e o
        resultado é stemizado pela mesma rota que gerou `texto_bm25`. Remover
        depois exigiria manter uma lista de radicais, que é ilegível.

        A comparação ignora acento, e isso foi um defeito medido em 14/11: a
        mesma pergunta escrita "reclamações sobre os Correios" e "reclamacoes
        sobre os Correios" produzia recuperação diferente, porque só a primeira
        casava com a lista de moldura. Quem digita pergunta omite acento, e o
        sistema não pode depender disso.
        """
        limpa = " ".join(
            p for p in pergunta.lower().split()
            if self._sem_acento(p.strip(".,;:?!")) not in _MOLDURA_SEM_ACENTO
        )
        tokens = pipeline_lexical_bm25([limpa or pergunta])[0]
        return tokens

    def _perna_lexical(self, pergunta: str, n: int):
        """BM25 com termo obrigatório.

        O termo de maior IDF da consulta passa a ser exigido no documento.
        Sem isso, "reclamações sobre os Correios" devolvia "Sem reclamações":
        um documento de 2 palavras que casa o termo banal e ganha pela
        normalização de comprimento. Com a exigência, só concorre quem
        contém de fato o termo que distingue a pergunta.
        """
        tokens = self._tokens_consulta(pergunta)
        if not tokens:
            return np.array([], dtype=int), np.array([]), tokens, None

        escores = self.bm25.get_scores(tokens)

        # IDF vem do próprio índice: é a raridade medida no corpus, não um
        # julgamento meu sobre qual palavra importa.
        idfs = {t: self.bm25.idf.get(t, 0.0) for t in tokens}
        obrigatorio = max(idfs, key=idfs.get) if idfs else None
        if obrigatorio is not None and idfs[obrigatorio] >= IDF_MINIMO_OBRIGATORIO:
            tem = self._docs_com(obrigatorio)
            if tem.any():
                escores = np.where(tem, escores, 0.0)
            else:
                obrigatorio = None  # termo ausente do corpus: não filtra nada
        else:
            obrigatorio = None

        pos = np.argsort(escores)[::-1][:n]
        pos = pos[escores[pos] > 0]
        return pos, escores[pos], tokens, obrigatorio

    # ------------------------------------------------------------------
    # Fusão
    # ------------------------------------------------------------------
    @staticmethod
    def _rrf(rankings: list[np.ndarray]) -> dict[int, float]:
        fundido: dict[int, float] = {}
        for ranking in rankings:
            for posicao, doc in enumerate(ranking):
                fundido[int(doc)] = fundido.get(int(doc), 0.0) + 1.0 / (RRF_K + posicao + 1)
        return fundido

    def buscar(self, pergunta: str, k: int = 5) -> Resultado:
        pos_vet, sim_vet = self._perna_vetorial(pergunta, CANDIDATOS_POR_PERNA)
        pos_lex, esc_lex, tokens, obrigatorio = self._perna_lexical(
            pergunta, CANDIDATOS_POR_PERNA
        )

        fundido = self._rrf([pos_vet, pos_lex])
        melhores = sorted(fundido, key=fundido.get, reverse=True)[:k]

        # A similaridade de cosseno é o que o portão lê, então ela precisa
        # existir para todo documento fundido — inclusive os que só o BM25
        # trouxe e que, por isso, não têm escore vetorial ainda.
        cos = dict(zip(pos_vet.tolist(), sim_vet.tolist()))
        faltantes = [p for p in melhores if p not in cos]
        if faltantes:
            vetor = self.modelo.encode(
                [normalizar_para_indexacao(pergunta)]
            ).astype("float32")
            self._faiss.normalize_L2(vetor)
            recon = np.vstack([self.indice.reconstruct(int(p)) for p in faltantes])
            for p, s in zip(faltantes, (recon @ vetor[0]).tolist()):
                cos[p] = s

        docs = self.mapa.iloc[melhores].copy()
        docs["similaridade"] = [cos[p] for p in melhores]
        docs["rrf"] = [fundido[p] for p in melhores]
        docs["so_lexical"] = [p not in set(pos_vet.tolist()) for p in melhores]
        docs = docs.sort_values("rrf", ascending=False).reset_index(drop=True)

        # Cobertura de vocabulário: em quantos documentos vive o termo central
        # da pergunta. Já está em cache de quando a perna lexical filtrou.
        df_assunto = (int(self._docs_com(obrigatorio).sum())
                      if obrigatorio else 0)

        suficiente, motivo = self._avaliar(docs, tokens, df_assunto)
        return Resultado(
            pergunta=pergunta,
            documentos=docs,
            suficiente=suficiente,
            motivo=motivo,
            melhor_similaridade=float(docs.similaridade.max()) if len(docs) else 0.0,
            diagnostico={
                "tokens_bm25": tokens,
                "termo_obrigatorio": obrigatorio,
                "df_assunto": df_assunto,
                "candidatos_vetorial": len(pos_vet),
                "candidatos_lexical": len(pos_lex),
                "sobreposicao": len(set(pos_vet.tolist()) & set(pos_lex.tolist())),
                "promovidos_so_pelo_bm25": int(docs.so_lexical.sum()) if len(docs) else 0,
            },
        )

    # ------------------------------------------------------------------
    # O portão
    # ------------------------------------------------------------------
    @staticmethod
    def _avaliar(docs: pd.DataFrame, tokens: list[str],
                 df_assunto: int = 0) -> tuple[bool, str]:
        """Decide se há evidência suficiente. Duas vias, em OU.

        Via semântica: a MEDIANA do cosseno do top-k. Mediana e não máximo,
        porque o máximo premia o acerto isolado que uma palavra em comum
        produz — ver LIMIAR_SUFICIENCIA.

        Via lexical: o termo central da pergunta aparece em documentos
        suficientes para ser assunto do corpus — ver DF_MINIMO_ASSUNTO.

        As duas são independentes e cada uma cobre a cegueira da outra, pelo
        mesmo princípio das duas pernas da busca. A semântica não distingue
        "Brasil" de review de toalha; a lexical não entende paráfrase.

        O filtro de anedota (MIN_DOCS_SUFICIENTES) vale para as duas vias: não
        importa por onde a pergunta passou, uma única review parecida não é
        evidência.
        """
        if docs.empty:
            return False, "nenhum documento recuperado"

        mediana = float(docs.similaridade.median())
        citaveis = int((docs.similaridade >= LIMIAR_DOCUMENTO).sum())
        por_semantica = mediana >= LIMIAR_SUFICIENCIA
        por_vocabulario = df_assunto >= DF_MINIMO_ASSUNTO

        if not (por_semantica or por_vocabulario):
            return False, (
                f"nenhuma via passou: mediana do top-{len(docs)} em {mediana:.3f} "
                f"(< {LIMIAR_SUFICIENCIA}) e termo central em {df_assunto} documentos "
                f"(< {DF_MINIMO_ASSUNTO}) — pergunta provavelmente fora do corpus"
            )
        if citaveis < MIN_DOCS_SUFICIENTES:
            return False, (f"evidência passou mas só {citaveis} documento citável; "
                           f"mínimo {MIN_DOCS_SUFICIENTES} para não citar anedota")

        via = ("semântica e vocabulário" if por_semantica and por_vocabulario
               else "semântica" if por_semantica else "vocabulário")
        return True, (f"passou por {via}: mediana {mediana:.3f}, termo central em "
                      f"{df_assunto} documentos, {citaveis} citáveis")


# ----------------------------------------------------------------------
# Verificação manual. Cada pergunta abaixo testa uma coisa diferente.
# ----------------------------------------------------------------------
PERGUNTAS_TESTE = [
    # paráfrase: a pergunta não usa as palavras da review
    ("o que os clientes dizem sobre demora na entrega?", "paráfrase"),
    # termo exato: BM25 deve carregar
    ("reclamações sobre os Correios", "termo exato"),
    # negação: o teste da decisão central do projeto
    ("quem não recebeu o produto?", "negação"),
    # específico e presente
    ("produto chegou com a caixa amassada", "específico"),
    # FORA do corpus: o portão tem que fechar
    ("qual a política de troca e devolução da Olist?", "fora do corpus"),
    ("quem é o presidente do Brasil?", "fora do corpus"),
]


def main() -> None:
    busca = BuscaHibrida()
    print(f"índice: {busca.indice.ntotal:,} vetores | limiar {LIMIAR_SUFICIENCIA}\n")
    for pergunta, proposito in PERGUNTAS_TESTE:
        r = busca.buscar(pergunta, k=5)
        print(f"--- {proposito} " + "-" * (58 - len(proposito)))
        print(r)
        print(f"  diagnóstico: sobreposição {r.diagnostico['sobreposicao']}"
              f"/{CANDIDATOS_POR_PERNA}, só-BM25 "
              f"{r.diagnostico['promovidos_so_pelo_bm25']}, "
              f"obrigatório {r.diagnostico['termo_obrigatorio']!r}, "
              f"tokens {r.diagnostico['tokens_bm25']}\n")


if __name__ == "__main__":
    main()
