# Enunciado — Tech Challenge Fase 4

**Título:** Voice of Customer Intelligence com RAG
**Dados:** Brazilian E-Commerce Public Dataset by Olist (Kaggle), ~100 mil pedidos de 2016 a 2018, foco na tabela de avaliações.
**Formato:** em grupo. Vale 90% da nota da fase.
**Período de entrega:** 09/09/2026 a 17/11/2026.
**Bônus:** +10 pontos concluindo a trilha *Databricks Generative AI Engineering Pathway* (enviar certificado ou print como "Completed").

## Cenário
Gestores fazem perguntas em linguagem natural sobre as avaliações, por exemplo:
- quais os principais problemas relatados;
- reclamações sobre entrega;
- o que os clientes mais elogiam;
- padrões de insatisfação;
- quais avaliações são evidência de um insight.

## Fluxo esperado
pergunta → processamento da pergunta → embeddings → recuperação das avaliações → construção do contexto → geração com LLM → resposta fundamentada.

Tecnologias livres, **mas toda escolha precisa ser justificada**.

## Requisitos
1. **Base de conhecimento:** justificar seleção dos dados, limpeza, tratamento de valores ausentes, organização dos documentos, criação dos embeddings, armazenamento e indexação.
2. **Pipeline RAG:** mostrar transformação da pergunta, recuperação semântica, seleção dos mais relevantes, montagem do contexto e geração.
3. **Respostas fundamentadas:** mostrar as avaliações usadas como evidência. A avaliação olha principalmente a recuperação e a coerência com as evidências.
4. **Limites:** reconhecer falta de evidência, evidência insuficiente e pergunta fora do escopo.

**Desafio adicional (opcional):** usar metadados (nota, período, pedido, entrega, categoria do produto), justificando como melhoram a recuperação.

## Entregáveis
- **Repositório público no GitHub** com: código, instruções de instalação e execução, README, arquitetura, tecnologias, estratégia de RAG, preparação da base, justificativas, exemplos de perguntas e respostas, reprodutibilidade.
- **Vídeo de até 7 minutos** cobrindo: problema, arquitetura, estratégia de RAG, recuperação, demonstração, exemplos, evidências, desafios e decisões.

## Critérios de avaliação
- recuperar conteúdo relevante;
- montar contexto adequado;
- respostas coerentes com as evidências;
- evitar informação sem suporte;
- rastreabilidade dos insights;
- reconhecer quando não há informação suficiente.

## Conteúdo da fase que compõe o TC
NLP (aulas 1–6), LangChain (aulas 1–5), IA Generativa (aulas 1–6: Transformer, RAG, prompts, agentes, multimodais, ética/viés/segurança).
