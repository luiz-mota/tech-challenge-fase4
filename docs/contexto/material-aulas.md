# Material das aulas útil para o TC

Repo do curso: `C:\pos-tech\POSTECH_AI_SCIENTIST\Fase 4\`

---

## IA Generativa

**Aula 2 — RAG** · `IA Generativa/aula02-rag/handson/aula02_handson.ipynb` · **base principal do projeto**
- Pipeline completo: ingestão/chunking, embeddings + índice FAISS, BM25 + fusão híbrida, geração com grounding/citação/recusa, avaliação recall@k/latência/custo.
- 8 etapas: 5 offline (ingestão, tratamento, chunking, embeddings, indexação), 3 online (recuperação, re-rank, geração).
- Metadados viajam com o documento e permitem filtrar **antes** da busca (precisão, custo, segurança).
- RRF: `score = Σ 1/(k + posição)`, k=60. É fusão por posição, não re-rank com modelo.
- Recall@k = |relevantes ∩ top-k| / |relevantes|. "Sem recall não há grounding."
- Faithfulness = afirmações sustentadas pelo contexto / total de afirmações.
- Geração com grounding: responder só com o contexto, recusar explicitamente, citar fontes.

**Aula 3 — Engenharia de prompts** · `aula03...`
- Anatomia: instrução + contexto + exemplos + política de saída. Prompt é artefato versionável.
- Separar instruções de dados com delimitadores ou tags XML.
- Persona específica ("analista de VoC que identifica problemas recorrentes"), não genérica.
- Contexto longo: documentos no topo, pergunta no fim (mitiga "lost in the middle").
- Few-shot com casos difíceis (ironia, review mista). CoT só onde há raciocínio.
- Temperatura baixa para extração e classificação. Mexer em temperatura **ou** top-p.

**Aula 4 — Agentes** · `aula04_handson.ipynb` (ReAct sem framework, schemas pydantic, limite de iterações, trace)
- Workflow (controle no código) × agente (controle parcial no modelo). Para o TC, **workflow fixo** é mais fácil de justificar.
- Se usar roteador/agente: tools só leitura, com schema; argumento inválido não executa.

**Aula 5 — Multimodal:** não se aplica (reviews só têm texto). No máximo citar como evolução futura (fotos de produto avariado).

**Aula 6 — Ética, viés, segurança** · `aula06...` (detector de PII pt-BR, detector de prompt injection, política como código, pipeline defensivo)
- Reviews são dado não confiável → prompt injection indireta.
- Mascarar PII nos comentários.
- Guardrail de saída: exigir `review_id`, recusar sem evidência.
- Discutir viés da amostra e LGPD/minimização no README.

---

## Processamento de Linguagem Natural

`Processamento de Linguagem Natural/aula_01..06/` (cada uma com `aulaNN_acompanhamento.ipynb` + um .py por slide)

**Aula 2 — Pré-processamento**
- Duas versões de limpeza: agressiva para BoW/TF-IDF/tópicos, leve para modelos contextuais. Documentar no README.
- Tratar NaN/não-string, HTML, URL, contrações PT-BR ("vc", "tbm", "pq", "n" → "não").
- **Não remover negações.** Exemplo do PDF: `pipeline_nlp("Não gostei, veio quebrado e atrasado.")` → `['gost','vei','quebr','atras']`, o "não" sumiu.

**Aula 3 — BoW e TF-IDF**
- Perna lexical da busca híbrida. N-grams (1,2) pegam "não chegou", "não recebi".
- Top termos TF-IDF por grupo (notas 1–2 × 4–5) dão evidência quantitativa de temas.
- `TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.85, sublinear_tf=True)`.

**Aula 4 — Embeddings**
- Embedding de **sentença** por review (Sentence-BERT multilíngue, BERTimbau ou API).
- Subword/contextual aguenta erro de digitação e gíria.
- t-SNE/UMAP dos embeddings como visualização no README/vídeo.

**Aula 5 — Classificadores clássicos**
- `review_score` como rótulo (1–2 negativo, 4–5 positivo).
- TF-IDF + LogisticRegression (`class_weight="balanced"`) para sentimento/tema → metadado e **contagem**.
- Reportar macro-F1 e matriz de confusão (base desbalanceada: muito mais nota 5 que nota 1).
- `Pipeline` do sklearn evita vazamento (vetorizador ajustado só no treino).

**Aula 6 — Tópicos** · **peça-chave para perguntas de frequência**
- BERTopic: Sentence-BERT → UMAP → HDBSCAN → c-TF-IDF. Pré-processamento mínimo. Tópico −1 = outliers.
- `BERTopic(embedding_model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"), language="multilingual", min_topic_size=20, nr_topics="auto")`.
- Contar documentos por tópico (geral e por faixa de nota), guardar o tópico como metadado.
- Visualizações (`visualize_topics`, `visualize_barchart`) servem para README e vídeo.
- Justificativa BERTopic × LDA: corpus de textos curtos em PT → BERTopic com embedding multilíngue.
- **O notebook da aula 6 avisa que LDA/BERTopic não rodam no ambiente do curso.** Testar instalação cedo.

---

## LangChain

`LangChain para aplicações de LLM/AulaNN/videoNN/main.py` + `pyproject.toml`

- **Aula 1:** `ChatPromptTemplate | llm | StrOutputParser`; `with_structured_output(ModeloPydantic)` para contrato de saída.
- **Aula 2:** `RunnablePassthrough` mantém a pergunta viva ao injetar contexto; roteamento por classificador com `with_structured_output` + validador Pydantic + fallback (padrão do roteador evidência × frequência); `.with_fallbacks()` entre provedores.
- **Aula 3.2:** FAISS + multi-query (gera variações da pergunta, une resultados sem repetir); prompt "responda só com o contexto; se não souber, diga que não encontrou".
- **Aula 3.3:** memória em SQLite com `RunnableWithMessageHistory` (só se houver chat multi-turno).
- **Aula 4.4:** API FastAPI com `create_agent`, auth por `X-API-Key`, contratos Pydantic de request/response, `/health`.
- **Aula 5.2:** LangGraph com checkpoint e human-in-the-loop (opcional, só se virar agente).
