# Cruzamento: requisito × decisão proposta × justificativa

Status: **proposta**. As decisões confirmadas pelo grupo vão para `decisoes.md`.

---

## R1 — Base de conhecimento

**Seleção dos dados.** Join de `olist_order_reviews_dataset` + `olist_orders_dataset` + `olist_order_items_dataset` + `olist_products_dataset` + `product_category_name_translation`.

**Valores ausentes.** Boa parte das reviews tem só nota, sem título nem comentário.
- Reviews **com texto** viram documentos do índice.
- Reviews **sem texto** não são descartadas: entram na camada de contagem e de distribuição de notas.
- Justificativa: uma review sem texto não serve como evidência citável, mas continua sendo sinal estatístico.

**Checagens obrigatórias antes de indexar:**
- unicidade de `review_id` (verificar se há IDs repetidos entre pedidos; se houver, definir chave composta `review_id + order_id`);
- review ligada a pedido com vários itens/categorias → decidir como representar a categoria (lista ou categoria principal);
- datas nulas de entrega (pedidos não entregues/cancelados).

**Organização dos documentos.** Uma review = um documento (título + mensagem concatenados). Sem chunking: textos curtos já são a unidade semântica certa, e cortá-los só gera ruído (IA Gen aula 2).

**Limpeza (duas versões, NLP aula 2):**
- *Leve* (embeddings e LLM): normalizar espaços, encoding, HTML/URL. Nada de stopwords nem stemming.
- *Agressiva* (BM25 e tópicos): lowercase, contrações ("vc", "tbm", "pq", "n" → "não"), stopwords, lematização.
- Em **nenhuma** das duas remover negações.

**Embeddings.** Modelo de sentença multilíngue com bom PT-BR (ex.: `paraphrase-multilingual-MiniLM-L12-v2`, ou maior se o tempo permitir). Justificativa: reviews em PT informal, com gíria e erro de digitação. Embedding de palavra (Word2Vec/FastText) é baseline barato, mas perde contexto e negação.

**Armazenamento e indexação.** FAISS local + índice BM25. Justificativa: volume pequeno, custo zero, reprodutível. Serviço gerenciado (Azure AI Search, Bedrock KB, Vertex RAG Engine) citado como caminho de produção.

---

## R2 — Pipeline RAG

**Transformação da pergunta.** Extrair filtros de metadado (período, nota, categoria) + reformular para busca. Multi-query (LangChain 3.2) como alternativa ou complemento.
> Ex.: "reclamações de entrega em 2018" → busca semântica por atraso/não chegou/demora + filtro `ano=2018`.

**Recuperação.** Busca híbrida: BM25 + vetorial, fusão por RRF (k=60).
- BM25 pega termos exatos (transportadora, "Correios", "nota fiscal").
- Vetorial pega paráfrase ("demorou demais" ≈ "atrasou").
- Filtros de metadado aplicados **antes** da busca.

**Seleção.** Top-k pós-fusão. Re-rank com cross-encoder é opcional; se cortado, citar como melhoria futura.

**Montagem do contexto.** System prompt com persona específica (analista de Voice of Customer), instruções, política de saída. Reviews em tags XML com `review_id`, nota, data e categoria. Documentos no topo, pergunta no fim (IA Gen aula 3).

**Geração.** Temperatura 0–0.2. Saída estruturada com Pydantic via `with_structured_output`:
```
resposta: str
evidencias: list[review_id]
suficiencia: "alta" | "media" | "baixa"
```

---

## R3 — Respostas fundamentadas

- A saída estruturada obriga a citar `review_id`.
- **Guardrail de citação** (política como código, IA Gen aula 6): rejeita resposta sem ID válido ou com ID que não estava no contexto.
- A interface mostra o trecho, a nota e a data de cada review citada.

---

## R4 — Limites

| Caso | Exemplo | Tratamento |
|---|---|---|
| Fora do escopo | "qual a capital da França?" | Classificar a pergunta antes da busca e recusar |
| Sem evidência | tema que não aparece nas reviews | Score de recuperação abaixo do limiar → "não encontrei avaliações sobre isso", sem chamar o LLM |
| Evidência insuficiente | só 2 reviews relevantes | Responde, com `suficiencia="baixa"` e aviso explícito |

O limiar é calibrado no golden dataset, e o README explica como se chegou ao número.

---

## Armadilha central: perguntas de frequência

O RAG devolve as k reviews mais **parecidas** com a pergunta, não as mais **frequentes**. "Quais os principais problemas?" respondido só com RAG parece certo e é metodologicamente errado.

**Solução: camada de agregação offline.**
1. BERTopic (NLP aula 6) sobre todas as reviews com texto, para descobrir os temas e atribuir um a cada review. Plano B se o BERTopic não instalar: KMeans sobre os embeddings + top termos TF-IDF por cluster (NLP aula 3). Alternativa supervisionada: TF-IDF + LogisticRegression (NLP aula 5).
2. Tema gravado como metadado no índice.
3. O roteador classifica a pergunta como de evidência ou de frequência. Na de frequência: contagem por tema (geral ou filtrada) → top temas → busca híbrida por tema → LLM resume cada tema citando evidências.

---

## Desafio opcional: metadados

Metadados por review: `review_score`, data da review, data de compra, **atraso de entrega** (`order_delivered_customer_date − order_estimated_delivery_date`), categoria do produto, tema.

O atraso calculado é o metadado mais forte: permite validar se as reclamações de entrega batem com os pedidos que de fato atrasaram. É rastreabilidade de insight, um dos critérios.

---

## Avaliação (mostrar no README e no vídeo)
- Golden dataset com 15 a 20 perguntas, incluindo fora do escopo e sem evidência (RAGAS pode ajudar a gerar).
- Recuperação: Recall@k.
- Geração: faithfulness e answer relevancy (RAGAS).
- Operação: latência e custo por pergunta.

## README: pontos que quase ninguém lembra
- Prompt injection indireta: reviews são texto de terceiros.
- PII nos comentários: mascarado na ingestão.
- Viés da amostra: só comenta quem está muito satisfeito ou muito insatisfeito. Os "principais problemas" são os de quem reclama, não de toda a base.
- LGPD e minimização de dados.
- RAG × fine-tuning: informação que muda e precisa de fonte → RAG.
- Workflow × agente: workflow fixo é previsível, testável e suficiente para o escopo.

## Ordem sugerida (5 dias)
1. Join, filtro, checagens, índices FAISS + BM25 (partindo do `aula02_handson.ipynb`), teste do BERTopic.
2. Prompt estruturado, guardrail de citação, os três casos de recusa, roteador de frequência.
3. Golden dataset, RAGAS, README.
4. Vídeo de 7 minutos + folga.
