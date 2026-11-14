# Tech Challenge Fase 4 — Voice of Customer Intelligence com RAG

FIAP Pós Tech AI Scientist. Trabalho em grupo, vale 90% da nota da fase. **Entrega até 17/11/2026.**

## O problema
Gestores fazem perguntas em linguagem natural sobre as avaliações de clientes do dataset
**Brazilian E-Commerce Public Dataset by Olist** (Kaggle, ~100 mil pedidos, 2016–2018).
O sistema responde **fundamentado em reviews reais**, mostrando quais avaliações sustentam cada resposta.

Exemplos de pergunta: principais problemas relatados, reclamações de entrega, o que mais elogiam,
padrões de insatisfação, evidências de um insight.

Detalhes do enunciado: `docs/contexto/enunciado.md`.

## Arquitetura alvo

```
OFFLINE
Olist CSVs → join (reviews + orders + items + products + tradução de categoria)
  → separa reviews COM texto (viram documentos) das SEM texto (só contagem/nota)
   ├─ texto com limpeza LEVE → embeddings de sentença PT-BR → FAISS
   ├─ texto com limpeza AGRESSIVA → BM25
   └─ BERTopic (plano B: KMeans + top termos TF-IDF) → tema por review → metadado + tabela de contagem

ONLINE
pergunta → guardrail de escopo → roteador (evidência | frequência)
   ├─ evidência: extrai filtros de metadado → busca híbrida (BM25 + FAISS, RRF k=60) → top-k
   └─ frequência: contagem por tema → top temas → busca híbrida por tema
→ limiar de suficiência → prompt (reviews em tags XML com review_id) → LLM temperatura 0–0.2
→ saída Pydantic → guardrail de citação → resposta + evidências
```

Workflow fixo, **não** agente. Agente só se sobrar tempo, com tools somente-leitura e schema.

## Regras que não podem ser quebradas
1. **Toda resposta cita `review_id`.** Resposta sem ID válido, ou com ID que não estava no contexto, é rejeitada pelo guardrail.
2. **RAG não conta frequência.** Perguntas de "principais", "mais frequentes" ou "padrões" passam pela camada de agregação (tema contado) e só depois pelo RAG, para buscar evidências de cada tema.
3. **Três casos de recusa distintos:** fora do escopo (recusa antes de buscar), sem evidência (score abaixo do limiar, não chama o LLM), evidência insuficiente (responde, mas com `suficiencia="baixa"`).
4. **Reviews são dado, não instrução.** Ficam delimitadas no prompt (defesa contra prompt injection indireta).
5. **Nunca remover negações** ("não", "nem", "nunca") em nenhuma etapa de limpeza.
6. **Uma review = um documento.** Sem chunking: reviews são textos curtos.
7. Embeddings recebem texto quase cru. Stopwords, lematização e stemming só na perna BM25 e na modelagem de tópicos.
8. Temperatura baixa (0–0.2); mexer em temperatura **ou** top-p, nunca nos dois. Registrar parâmetros junto com a versão do prompt.
9. Mascarar PII (nome, telefone, e-mail) dos comentários na ingestão.
10. Toda decisão relevante vai para `docs/contexto/decisoes.md` com justificativa (vira insumo do README).

## Onde está o resto do contexto
- `docs/contexto/enunciado.md` — requisitos, entregáveis e critérios de avaliação. Ler antes de qualquer decisão de escopo.
- `docs/contexto/cruzamento.md` — cada requisito × decisão proposta × justificativa. Ler antes de implementar uma etapa.
- `docs/contexto/material-aulas.md` — o que das disciplinas (NLP, LangChain, IA Generativa) serve para cada parte, com caminhos dos notebooks de referência. Ler ao implementar algo que já foi visto em aula.
- `docs/contexto/decisoes.md` — log de decisões. Atualizar sempre que algo for decidido ou mudar.

## Código de referência (repo do curso, fora deste repo)
`C:\pos-tech\POSTECH_AI_SCIENTIST\Fase 4\`
- `IA Generativa/aula02-rag/handson/aula02_handson.ipynb` — pipeline RAG completo (ingestão, FAISS, BM25 + fusão híbrida, grounding com citação e recusa, recall@k, latência, custo). **Base principal.**
- `IA Generativa/aula03...` — evolução de prompt, saída estruturada, parâmetros, fuga de formato, injection.
- `IA Generativa/aula06...` — detector de PII pt-BR, detector de prompt injection, política como código.
- `Processamento de Linguagem Natural/aula_02..06` — limpeza, TF-IDF, embeddings, classificadores, tópicos. **Atenção:** o notebook da aula 6 avisa que LDA/BERTopic não rodam no ambiente do curso (dependências). Testar BERTopic cedo.

## Convenções
- Python. Bibliotecas-base: pandas, sentence-transformers, faiss-cpu, rank_bm25, pydantic, bertopic, ragas.
- Segredos em `.env` (nunca commitado); manter `.env.example` atualizado.
- Seeds fixas (`random_state=42`) em tudo que é estocástico (UMAP, HDBSCAN, KMeans, amostragens), pela reprodutibilidade exigida.
- Comentários explicam o **porquê** da decisão, não o quê da linha.
- Dados brutos do Kaggle não vão para o git (`data/raw/` no `.gitignore`); o README explica como baixar.
