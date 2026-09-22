"""Monta a base de conhecimento a partir dos CSVs brutos do Olist.

Requisito 1 do Tech Challenge. Cada filtro abaixo é uma decisão justificada:

  1. só reviews com `review_comment_message`  -> 41,3% das 99.224 avaliações
  2. dedupe por `review_id`                   -> 789 ids aparecem em 2+ linhas
  3. título concatenado à mensagem            -> recupera 1.737 reviews
  4. piso de 4 palavras                       -> corta "bom", "ok", "recomendo"
  5. um documento por review, sem chunking    -> campo truncado em 208 chars

Saída: data/processed/corpus.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.knowledge_base.limpeza import normalizar_para_indexacao  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"

MIN_PALAVRAS = 4


def _log(etapa: str, n: int, anterior: int | None = None) -> None:
    delta = "" if anterior is None else f"   ({n - anterior:+,})"
    print(f"  {etapa:<44} {n:>7,}{delta}")


def carregar_reviews() -> pd.DataFrame:
    print("\n[1] Seleção dos documentos")
    r = pd.read_csv(RAW / "olist_order_reviews_dataset.csv")
    _log("avaliações no arquivo bruto", len(r))

    # Decisão: ausência de texto não é valor faltante a imputar. Um vetor de
    # texto vazio não carrega semântica. As 58.247 sem texto continuam válidas
    # como denominador estatístico, mas fora do índice.
    r["mensagem"] = r.review_comment_message.fillna("").str.strip()
    r["titulo"] = r.review_comment_title.fillna("").str.strip()
    n = len(r)
    r = r[r.mensagem != ""].copy()
    _log("com review_comment_message", len(r), n)

    # `review_id` não é chave: 789 ids em 1.603 linhas, mesmo texto e mesma
    # nota, order_id diferente. Sem dedupe o retrieval devolveria o mesmo
    # depoimento duas vezes e o LLM leria como duas evidências independentes.
    n = len(r)
    r = r.drop_duplicates(subset="review_id", keep="first")
    _log("após dedupe por review_id", len(r), n)

    # Título só existe em 11,7% dos casos, mas em 1.737 é o único texto.
    r["texto_bruto"] = (r.titulo + ". " + r.mensagem).where(r.titulo != "", r.mensagem)
    return r


def normalizar(r: pd.DataFrame) -> pd.DataFrame:
    print("\n[2] Normalização e piso de comprimento")
    r["texto"] = r.texto_bruto.map(normalizar_para_indexacao)
    r["n_palavras"] = r.texto.str.split().str.len()

    n = len(r)
    r = r[r.texto.str.strip() != ""]
    if len(r) != n:
        _log("texto não vazio após normalizar", len(r), n)

    n = len(r)
    r = r[r.n_palavras >= MIN_PALAVRAS].copy()
    _log(f"com >= {MIN_PALAVRAS} palavras", len(r), n)
    return r


def juntar_metadados(r: pd.DataFrame) -> pd.DataFrame:
    """Metadados para recuperação filtrada (item opcional do enunciado)."""
    print("\n[3] Metadados")
    orders = pd.read_csv(RAW / "olist_orders_dataset.csv")
    customers = pd.read_csv(RAW / "olist_customers_dataset.csv")
    itens = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    produtos = pd.read_csv(RAW / "olist_products_dataset.csv")
    traducao = pd.read_csv(RAW / "product_category_name_translation.csv")

    df = r.merge(
        orders[[
            "order_id", "customer_id", "order_status",
            "order_purchase_timestamp", "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]],
        on="order_id", how="left",
    )
    df = df.merge(
        customers[["customer_id", "customer_state", "customer_city"]],
        on="customer_id", how="left",
    )

    # Categoria: 96,2% dos pedidos têm uma só. Guardo a primeira e o total,
    # para que a resposta possa avisar quando o pedido misturar categorias.
    cat = (
        itens.merge(produtos[["product_id", "product_category_name"]],
                    on="product_id", how="left")
        .merge(traducao, on="product_category_name", how="left")
        .groupby("order_id")
        .agg(categoria=("product_category_name", "first"),
             n_categorias=("product_category_name", "nunique"))
        .reset_index()
    )
    df = df.merge(cat, on="order_id", how="left")

    # Derivado: atraso de entrega. 9,2% do corpus — habilita a pergunta
    # "o que dizem os clientes que receberam atrasado?".
    entregue = pd.to_datetime(df.order_delivered_customer_date)
    previsto = pd.to_datetime(df.order_estimated_delivery_date)
    comprado = pd.to_datetime(df.order_purchase_timestamp)
    df["atraso_dias"] = (entregue - previsto).dt.days
    df["entrega_atrasada"] = df.atraso_dias > 0
    df["dias_ate_entrega"] = (entregue - comprado).dt.days
    df["data_review"] = pd.to_datetime(df.review_creation_date)
    df["ano_mes"] = df.data_review.dt.to_period("M").astype(str)

    _log("sem match em orders", int(df.order_status.isna().sum()))
    _log("sem categoria de produto", int(df.categoria.isna().sum()))
    _log("entregas atrasadas", int(df.entrega_atrasada.sum()))
    return df


def salvar(df: pd.DataFrame) -> Path:
    colunas = [
        "review_id", "order_id", "texto", "texto_bruto", "n_palavras",
        "review_score", "data_review", "ano_mes", "order_status",
        "entrega_atrasada", "atraso_dias", "dias_ate_entrega",
        "categoria", "n_categorias", "customer_state", "customer_city",
    ]
    df = df[colunas].reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    destino = PROCESSED / "corpus.parquet"
    df.to_parquet(destino, index=False)

    print("\n[4] Corpus final")
    _log("documentos", len(df))
    print(f"  {'palavras por documento (mediana / p90 / max)':<44} "
          f"{df.n_palavras.median():.0f} / {df.n_palavras.quantile(.9):.0f} "
          f"/ {df.n_palavras.max():.0f}")
    print(f"\n  distribuição por nota:")
    for nota, qtd in df.review_score.value_counts().sort_index().items():
        print(f"    {nota} estrela(s): {qtd:6,}  ({qtd / len(df) * 100:4.1f}%)")
    print(f"\n  gravado em {destino.relative_to(RAIZ)} "
          f"({destino.stat().st_size / 1e6:.1f} MB)")
    return destino


def main() -> pd.DataFrame:
    df = salvar(juntar_metadados(normalizar(carregar_reviews())))
    return df


if __name__ == "__main__":
    main()
