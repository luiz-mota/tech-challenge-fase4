"""Baixa o Brazilian E-Commerce Public Dataset by Olist para data/raw/.

Autenticação: kagglehub lê KAGGLE_USERNAME/KAGGLE_KEY do ambiente ou
~/.kaggle/kaggle.json. Token em https://www.kaggle.com/settings.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

DATASET = "olistbr/brazilian-ecommerce"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existentes = list(RAW_DIR.glob("*.csv"))
    if existentes and not force:
        print(f"{len(existentes)} CSVs já presentes em {RAW_DIR} — nada a fazer.")
        return RAW_DIR

    cache = Path(kagglehub.dataset_download(DATASET, force_download=force))
    for csv in cache.glob("*.csv"):
        shutil.copy2(csv, RAW_DIR / csv.name)

    copiados = sorted(RAW_DIR.glob("*.csv"))
    print(f"{len(copiados)} arquivos em {RAW_DIR}:")
    for csv in copiados:
        print(f"  {csv.name:55s} {csv.stat().st_size / 1e6:7.1f} MB")
    return RAW_DIR


if __name__ == "__main__":
    download()
