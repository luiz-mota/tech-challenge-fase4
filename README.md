# Voice of Customer Intelligence com RAG

Sistema de perguntas e respostas em linguagem natural sobre as avaliações de
clientes do **Brazilian E-Commerce Public Dataset by Olist**. Gestores perguntam
*"quais os principais problemas relatados nas entregas atrasadas?"* e o sistema
responde fundamentado nas reviews recuperadas, citando quais sustentam cada
afirmação — ou declarando que não há evidência suficiente.

Tech Challenge — Fase 4, Pós-Tech FIAP (AI Scientist).

> **Estado atual:** Requisito 1 (Construção da Base de Conhecimento) implementado
> até a etapa de seleção e preparação dos documentos. Embeddings, indexação,
> recuperação e geração estão em desenvolvimento — as seções correspondentes
> abaixo estão marcadas como pendentes.

---

## Sumário

- [O problema de negócio](#o-problema-de-negócio)
- [Instalação](#instalação)
- [Reprodução](#reprodução)
- [Construção da base de conhecimento](#construção-da-base-de-conhecimento)
- [Arquitetura](#arquitetura) *(pendente)*
- [Pipeline RAG](#pipeline-rag) *(pendente)*
- [Rastreabilidade e abstenção](#rastreabilidade-e-abstenção) *(pendente)*
- [Estrutura do repositório](#estrutura-do-repositório)

---

## O problema de negócio

O Olist acumula 99.224 avaliações entre outubro de 2016 e agosto de 2018. A nota
de 1 a 5 diz *o quanto* o cliente ficou insatisfeito; só o texto diz **por quê**.
Esse texto existe em 41,3% das avaliações e nunca foi lido em conjunto — é
volume demais para leitura manual e desestruturado demais para SQL.

A pergunta que o negócio faz não é "qual a nota média da categoria X", que um
`GROUP BY` responde. É "o que os clientes reclamam quando a entrega atrasa" —
uma pergunta sobre conteúdo, que exige recuperar os depoimentos semanticamente
relacionados e sintetizá-los **sem inventar**.

---

## Instalação

Requer Python 3.14 (testado em 3.14.0, Windows).

```bash
git clone https://github.com/luiz-mota/tech-challenge-fase4.git
cd tech-challenge-fase4

python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['stopwords','rslp','punkt','punkt_tab']]"
```

## Reprodução

```bash
python src/ingestion/download_olist.py        # baixa os 9 CSVs (~43 MB)
python src/knowledge_base/preparar_corpus.py  # gera data/processed/corpus.parquet
pytest tests/ -q                              # valida as decisões de limpeza
```

Os dados não são versionados: são reproduzíveis pelos comandos acima.

---

## Construção da base de conhecimento

### Seleção dos dados

A base do Olist tem nove tabelas. A tabela `olist_order_reviews_dataset` é a
única que contém texto escrito por clientes e por isso é a única fonte de
documentos. As demais entram apenas como **metadados** de filtro (seção
adiante), nunca como conteúdo indexado — descrições de produto e nomes de
categoria não são voz do cliente, e misturá-las ao índice faria o sistema
recuperar catálogo quando o gestor pergunta por reclamação.

### Tratamento de valores ausentes

| campo | ausentes | decisão |
|---|---:|---|
| `review_comment_message` | 58.247 (58,7%) | documento não é criado |
| `review_comment_title` | 87.656 (88,3%) | concatenado quando existe |
| `review_score` | 0 | — |
| `order_delivered_customer_date` | 1.720 no corpus (5,1%) | metadado fica nulo |

**Ausência de texto não é valor faltante a imputar.** Não existe imputação
honesta para uma opinião que o cliente não escreveu: preencher com a nota, com a
média da categoria ou com string vazia produziria um vetor sem significado, que
o retrieval devolveria como se fosse depoimento. As 58.247 avaliações sem texto
permanecem disponíveis como **denominador estatístico**, fora do índice.

O título existe em apenas 11,7% dos casos, mas em **1.737 avaliações ele é o
único texto**. É concatenado à mensagem (`"{título}. {mensagem}"`) em vez de
virar campo separado, porque título e corpo são a mesma opinião fragmentada pelo
formulário.

### O viés do corpus textual — restrição de projeto

Quem está insatisfeito escreve mais:

| nota | avaliações | escreveram texto | % |
|---:|---:|---:|---:|
| 1 | 11.424 | 8.744 | **76,5%** |
| 2 | 3.151 | 2.145 | 68,1% |
| 3 | 8.179 | 3.556 | 43,5% |
| 4 | 19.142 | 5.970 | 31,2% |
| 5 | 57.328 | 20.535 | 35,8% |

Nota 1 representa 11,5% da base completa e **24,5% do corpus indexado**. Toda
contagem feita sobre os documentos recuperados descreve, portanto, *quem
escreveu*, não *quem comprou*.

Isso não é corrigível por amostragem — reponderar destruiria a cobertura
temática. É tratado como **restrição explícita**: o sistema responde sobre
*temas relatados* e nunca sobre *proporção de clientes*, e a instrução consta do
prompt de geração.

### Limpeza e preparação dos textos

A decisão de maior impacto do projeto. O material do curso fornece um pipeline
canônico de pré-processamento — limpeza, remoção de stop words e stemização
RSLP. Aplicá-lo ao texto que vira embedding **inverteria o sentido do corpus**:

```
'Não gostei, veio quebrado e atrasado.'  ->  ['gost', 'vei', 'quebr', 'atras']
```

`não`, `nem`, `sem` e `mas` estão entre as 207 stop words do NLTK em português.
Após a remoção, *"Não chegou o produto"* e *"Chegou o produto"* produzem vetores
idênticos. **25,9% do corpus contém marcador de negação** — numa base cujo valor
está nas reclamações, esse pré-processamento é destrutivo.

O mesmo vale para o filtro de caracteres, que descarta dígitos:

```
ANTES : Comprei o produto dia 25 de fevereiro e hoje dia 29 de março não fora entregue
DEPOIS: comprei o produto dia de fevereiro e hoje dia de março não fora entregue
```

O atraso de 33 dias — o conteúdo inteiro da reclamação — desaparece. **10,05%
das reviews contêm dígito.**

A causa é conceitual, não um defeito do material: BoW e TF-IDF representam
**frequência**, e ali `não` é de fato ruído de alta frequência. Embeddings densos
representam **sentido**, e tudo que se apaga antes de gerá-los é uma declaração
de que aquilo não altera a distância entre dois textos. São objetivos opostos.

O projeto mantém as duas rotas, com destinos separados
([`src/knowledge_base/limpeza.py`](src/knowledge_base/limpeza.py)):

| | `normalizar_para_indexacao()` | `pipeline_nlp_analitico()` |
|---|---|---|
| destino | texto vetorizado (retrieval) | análise do corpus, BoW/TF-IDF |
| stop words | preserva | remove |
| stemização | não | RSLP |
| caixa / acentos / dígitos / emoji | preserva | descarta |
| representa | sentido | frequência |

**O que `normalizar_para_indexacao()` remove**, medido nas 40.950 reviews com
texto — apenas ruído de superfície:

| operação | reviews afetadas |
|---|---:|
| normalização Unicode NFC | — (acentos decompostos quebram a tokenização) |
| remoção de HTML | 0 |
| remoção de URL / e-mail | 2 |
| expansão de contrações (`pra`, `pq`, `n`, `vc`…) | 1.391 (3,40%) |
| colapso de pontuação repetida (`!!!!` → `!`) | 3.977 (9,71%) |

**O que preserva, e por quê:**

- **caixa** — 3,55% das reviews são inteiramente em maiúsculas; é ênfase, não ruído;
- **acentos** — 51,24% têm acento. O downstream é um tokenizer treinado em
  português: `coração` está no vocabulário, `coracao` fragmenta em subwords.
  A pergunta do material ("remover acentos?") é sensível ao downstream, e para
  retrieval denso a resposta é não;
- **dígitos** — 10,05%, e é neles que mora a evidência quantitativa;
- **emoji** — 0,67%, carregam polaridade;
- **negação** — 25,9%, pelo motivo acima.

### Organização dos documentos

```
99.224  avaliações no arquivo bruto
40.950  com review_comment_message          −58.274
40.641  após dedupe por review_id              −309
33.812  com >= 4 palavras após normalizar   −6.829
```

**Dedupe por `review_id`.** O campo não é chave: 789 ids aparecem em duas ou mais
linhas, com mesmo texto e mesma nota, ligados a `order_id` diferentes. Sem
dedupe, o mesmo depoimento é recuperado duas vezes e o LLM o lê como duas
evidências independentes — evidência inflada por artefato da base.

**Piso de 4 palavras.** 19% das mensagens têm até 3 palavras (`"bom"`,
`"recomendo"`, `"ok"`); 5.944 linhas são repetição literal de outra (`"muito
bom"` ocorre 472 vezes). Não sustentam nenhum insight e competem por vaga no
top-k da recuperação. O corte é parâmetro (`MIN_PALAVRAS`), fixado em 4 por
preservar reclamações curtas legítimas — *"chegou quebrado"*, *"produto errado"*
— que um corte em 6 eliminaria.

**Um documento por review, sem chunking.** O campo é truncado em 208 caracteres
na origem: a maior review do corpus tem 49 palavras, mediana de 11. Nenhuma
chega perto do limite de qualquer modelo de embedding, e fragmentar separaria o
problema da sua justificativa dentro da mesma frase.

**Corpus final: 33.812 documentos**, mediana de 11 palavras, máximo de 49.

### Metadados para recuperação filtrada

Todos os 33.812 documentos casam com `olist_orders_dataset`. Cada documento
carrega:

| campo | cobertura | habilita |
|---|---|---|
| `review_score` | 100% | "o que dizem os detratores?" |
| `data_review`, `ano_mes` | 100% | recortes temporais, tendência |
| `order_status` | 100% | separar cancelado de entregue |
| `entrega_atrasada`, `atraso_dias` | 94,9% | **3.433 docs** de entrega atrasada |
| `dias_ate_entrega` | 94,9% | correlação prazo × insatisfação |
| `categoria`, `n_categorias` | 97,2% | recorte por linha de produto |
| `customer_state`, `customer_city` | 100% | recorte regional |

Metadado melhora a recuperação porque converte restrição factual em filtro
exato, em vez de deixá-la para a busca semântica. *"Reclamações de atraso em
móveis no Nordeste em 2018"* tem três condições verificáveis (`entrega_atrasada`,
`categoria`, `customer_state`, `ano_mes`) e uma semântica (o teor da reclamação).
Empurrar as factuais para o filtro libera o top-k inteiro para discriminar
conteúdo — sem isso, boa parte das vagas seria gasta com documentos do tema
certo e do recorte errado.

`atraso_dias` é derivado — diferença entre entrega real e data estimada — e não
existe como coluna na base original.

---

## Arquitetura

> **Pendente.** Modelo de embedding, vector store e estratégia de recuperação
> ainda não definidos. Cada escolha será justificada tecnicamente nesta seção.

## Pipeline RAG

> **Pendente.** Transformação da pergunta → recuperação semântica → seleção do
> conteúdo relevante → construção do contexto → geração.

## Rastreabilidade e abstenção

> **Pendente.** Exibição das reviews que sustentam cada insight e critérios de
> abstenção (nada relevante recuperado / evidência insuficiente / pergunta fora
> de escopo).

---

## Estrutura do repositório

```
tech-challenge-fase4/
├── data/
│   ├── raw/                        CSVs do Olist (não versionado)
│   └── processed/corpus.parquet    base de conhecimento (não versionado)
├── src/
│   ├── ingestion/
│   │   └── download_olist.py       download reproduzível via kagglehub
│   └── knowledge_base/
│       ├── limpeza.py              as duas rotas de normalização
│       └── preparar_corpus.py      seleção, dedupe, metadados
├── tests/
│   └── test_limpeza.py             trava as decisões de limpeza
├── notebooks/                      análise exploratória
├── reports/                        figuras e resultados
└── requirements.txt
```

### Testes

`pytest tests/ -q` — 9 testes. Dois deles fixam a tese central:

```python
test_negacao_sobrevive()                    # "Não chegou" != "Chegou"   (indexação)
test_pipeline_analitico_perde_a_negacao()   # "Não chegou" == "Chegou"   (análise)
```

O segundo afirma que o pipeline analítico **perde** a negação, de propósito. Se
a limpeza de indexação for algum dia "melhorada" com remoção de stop words, o
primeiro teste falha e aponta o motivo.

---

## Tecnologias

| camada | escolha | justificativa |
|---|---|---|
| ingestão | `kagglehub` | download versionado e reproduzível, sem credencial |
| dados | `pandas` + Parquet | tipagem preservada; 6,1 MB contra 43 MB de CSV |
| NLP | `nltk` (stopwords pt, RSLP) | rota analítica |
| testes | `pytest` | decisões de limpeza como comportamento executável |
| embeddings / vector store / LLM | *a definir* | — |
