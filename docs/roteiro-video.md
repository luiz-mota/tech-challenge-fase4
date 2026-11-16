# Roteiro do vídeo — 5 minutos

Deck pronto para apresentar: **https://claude.ai/artifact/PG7pw411RDxAPMiYKob8hk**
(as falas também estão nas notas de cada slide, no modo apresentador).

**12 slides · 5 min 25 s no cronômetro.** O enunciado dá até 7 minutos, então há
folga. Se quiser fechar em 5:00 exatos, corte o slide 6 — está marcado abaixo.

Os nove tópicos obrigatórios do enunciado estão todos cobertos:

| tópico exigido | slide |
|---|---|
| problema | 2 |
| arquitetura | 3 |
| estratégia de RAG | 3, 5 |
| recuperação | 3, 11 |
| demonstração | 9 |
| exemplos | 7, 9 |
| evidências | 9 |
| desafios | 4, 5, 11 |
| decisões | 4, 5, 8, 11 |

---

## Antes de gravar

1. **Rode a demo uma vez e deixe o terminal aberto.** A primeira resposta leva
   ~20 s porque carrega o modelo; a segunda leva ~20 s também. Se for gravar a
   tela ao vivo, 20 s de silêncio custam 7% do vídeo inteiro.

   ```bash
   python app.py -p "o que dizem os clientes sobre a caixa amassada?" --fontes
   ```

   O slide 9 já tem essa saída impressa. **A recomendação é usar o slide** e
   mencionar que é saída real — fica no tempo e não depende da máquina.

2. **Teste o áudio antes.** Mais vídeo é reprovado por áudio ruim que por
   conteúdo.

3. **Fale olhando para a câmera no slide 1 e no 12.** No meio, pode ler.

---

## O roteiro, slide a slide

### 1 · Capa — 5 s

> Boa tarde. Vou mostrar um sistema que responde perguntas de gestor sobre as
> avaliações de clientes da Olist. A frase que importa é a segunda: toda
> resposta vem com a review que a sustenta, ou com uma recusa.

---

### 2 · O problema — 25 s

> São 99 mil avaliações. A nota de 1 a 5 diz **o quanto** o cliente ficou
> insatisfeito; só o texto diz **por quê**. Esse texto existe em 43% das
> avaliações e nunca foi lido em conjunto — é volume demais para leitura manual
> e desestruturado demais para SQL.
>
> A distinção que define o projeto está na última linha: *"qual a nota média da
> categoria X"* é um `GROUP BY`. *"O que os clientes reclamam quando a entrega
> atrasa"* é uma pergunta sobre **conteúdo** — e é aí que o RAG entra.

---

### 3 · Arquitetura — 40 s

> A arquitetura tem três rotas, e essa é a decisão de desenho mais importante.
>
> A rota de evidência é o RAG clássico: busca híbrida BM25 mais FAISS, fusão por
> RRF com k=60, portão de suficiência, modelo local, guardrail de citação.
>
> Mas reparem nas outras duas: elas **não passam pelo LLM** e respondem em 100
> milissegundos. Explico na próxima por que existem.
>
> O roteador é de **regra, não de modelo**: erra de forma previsível, e na
> dúvida manda para a rota que sabe abster. Tudo roda local, sem chave de API.

---

### 4 · A negação — 30 s

> Primeira decisão técnica. O pipeline de pré-processamento ensinado no curso
> remove stop words — e *"não"*, *"nem"*, *"sem"* e *"mas"* **são** stop words
> do NLTK.
>
> Olhem o exemplo: *"Não gostei, veio quebrado e atrasado"* vira quatro
> radicais, e o "não" desapareceu. Depois disso, *"não chegou o produto"* e
> *"chegou o produto"* produzem **vetores idênticos**. Numa base cujo valor
> inteiro está nas reclamações, isso é destrutivo: 25,9% do corpus tem negação.
>
> E não é defeito da aula — é que BoW representa **frequência**, onde o "não" é
> mesmo ruído, e embedding representa **sentido**. Objetivos opostos. Por isso o
> projeto tem **três** rotas de limpeza em vez de uma.

---

### 5 · Frequência ≠ similaridade — 30 s

> Segunda decisão, e é a armadilha central do desafio.
>
> O RAG devolve as k reviews mais **parecidas** com a pergunta, não as mais
> **frequentes**. Se eu mandasse *"quais os principais problemas"* para o RAG,
> ele traria cinco reviews que contêm a palavra "problema" — parece uma resposta
> certa e é metodologicamente errado.
>
> Por isso existe uma camada de contagem separada: BERTopic agrupando em 101
> temas, que eu consolidei à mão em 16 macro-temas. E a contagem sai sempre com
> o denominador: *"32% dos clientes reclamam de entrega"* seria **falso**;
> *"32% dos relatos de problema são de entrega"* é **verdade**.

---

### 6 · A distribuição dos temas — 25 s · **CORTE AQUI SE PRECISAR**

> Esta é a resposta da rota de frequência, e sai em 101 milissegundos. "Produto
> não recebido" lidera com 17,1% dos relatos.
>
> Reparem numa coisa de propósito: o **denominador está dentro da figura** —
> 13.166 avaliações com problema identificado, não as 99 mil da base. Gráfico de
> frequência sem denominador é exatamente o erro que essa camada existe para
> evitar.

*Cortando este slide, o vídeo fecha em 5:00. O slide 7 sustenta o argumento
sozinho — mas aí vale dizer "17,1% são de produto não recebido" na transição.*

---

### 7 · O tema distingue o que a nota não distingue — 35 s · **O MAIS IMPORTANTE**

> Esta é a figura que eu defenderia primeiro numa arguição.
>
> Os temas foram formados a partir do **texto**. O eixo que vocês estão vendo —
> o percentual de pedidos que de fato atrasaram — vem das **datas do pedido**. É
> informação independente, que não entrou na formação dos temas. E ela separa os
> grupos em dois blocos limpos.
>
> Olhem o detalhe que mais importa: *"produto não recebido"* tem nota média 1,8 e
> *"quantidade errada"* tem nota 2,0. **Pela nota são indistinguíveis. Pelo
> atraso real estão a dez vezes de distância.**
>
> Ou seja: a camada de temas acrescenta um eixo que a nota não tem. Se a minha
> leitura estivesse errada, essa coluna estaria embaralhada.

---

### 8 · Quatro camadas de abstenção — 30 s

> O requisito 4 pede que o sistema reconheça quando não sabe. São quatro
> camadas, e a coluna da direita é o ponto: **só uma depende do modelo
> obedecer.**
>
> A camada zero bloqueia pergunta com CPF antes de buscar qualquer coisa. A
> camada 1 é o portão: se a evidência é fraca, o LLM **nem é chamado**. A camada
> 3 confere por código se o `review_id` citado estava mesmo no contexto — se não
> estava, a resposta vira abstenção.
>
> O curso põe a recusa só no prompt; eu divergi, porque o LLM não pode recusar o
> que ninguém lhe disse que é fraco.

---

### 9 · Demonstração — 45 s

> Esta é a saída **real**, não uma maquete.
>
> Pergunto sobre a caixa amassada. O sistema mostra a rota que escolheu,
> responde em duas linhas, cita cinco reviews pelo id — e acrescenta uma
> **ressalva de evidência fraca**.
>
> E a ressalva está **certa**: das cinco reviews recuperadas, duas são nota 4 e
> 5, de gente que recebeu a caixa amassada e o produto intacto. O sistema está
> dizendo *"existe base, mas é fraca"* em vez de afirmar no mesmo tom de quando
> há folga.
>
> Embaixo, cada id abre o depoimento inteiro com data, nota, categoria e estado.
> É isso que torna o insight rastreável.

---

### 10 · Avaliação — 30 s

> Recall@5 de 10 em 12, com gabarito derivado dos temas. O portão foi calibrado
> em 22 perguntas e validado em **16 perguntas novas**, que nunca participaram
> da escolha do corte: 95% na calibração, 100% na validação. E invenção mediana
> **zero** nas respostas geradas.
>
> Sobre esse último: eu **não usei RAGAS**, e quero que fique claro como decisão
> e não como falta. RAGAS precisa de um LLM juiz, e usar o Qwen de 0,5 bilhão
> para julgar o próprio Qwen de 0,5 bilhão mediria a concordância de um modelo
> fraco consigo mesmo. Então medi **ancoragem lexical**: mais pobre, e
> verificável.

---

### 11 · Limites e divergências — 25 s

> Fecho com o que **não** funciona, porque vale mais que a lista do que funciona.
>
> O viés do corpus: só escreve quem está nos extremos, então a contagem descreve
> **quem escreveu**, nunca quem comprou — e isso sai junto de toda resposta.
>
> O stemizador separa substantivo de verbo: *"cobrança"* gera um radical que não
> existe em documento nenhum, enquanto *"cobrar"* existe em 200. Foi justamente
> esse defeito que **provou que a busca híbrida não é redundância**, porque só a
> perna vetorial salva esses casos.
>
> Divergi do curso em três pontos, todos declarados: sem chunking, sem LangChain
> e sem RAGAS. São 43 decisões registradas com a alternativa descartada e o
> custo assumido.

---

### 12 · Fecho — 10 s

> Termino com a ideia que eu levaria para qualquer RAG em produção. **A correção
> deste sistema não depende da qualidade do modelo:** o guardrail compara
> strings, a contagem vem do pandas, o portão vem do cosseno. O modelo só redige
> a frase final.
>
> É por isso que roda num ultrabook sem GPU com meio bilhão de parâmetros: um
> modelo pequeno escreve pior, **não cita falso**. Obrigado.

---

## Se a banca perguntar depois

| pergunta provável | resposta curta |
|---|---|
| *"Por que não usou LangChain?"* | A abstração esconde as peças que estão em avaliação — RRF, limiar, montagem do contexto. E portão, termo obrigatório e guardrail não existem no framework: teria de escrever à mão de qualquer jeito. Está registrado em `decisoes.md`. |
| *"Por que um modelo tão pequeno?"* | Restrição de memória medida: 0,2 GB livres. E a troca não afeta a correção — o guardrail é comparação de string. O 1,5B é selecionável por variável de ambiente. |
| *"Como sabe que não alucina?"* | Não sei que nunca alucina; sei que **não cita falso**, porque isso é conferido por código. E medi invenção lexical: mediana 0%. Os limites da métrica estão declarados. |
| *"O filtro de metadado funciona?"* | Na rota de contagem sim — categoria, UF, mês e atraso. Na rota de evidência **ainda não**, e está registrado como o próximo passo de maior retorno. |
| *"Por que BERTopic e não LDA?"* | Corpus de textos curtos em português. LDA precisa de documento longo para estimar a mistura de tópicos; a mediana aqui é 10 palavras. |
| *"E se o corpus crescer 100×?"* | FAISS `IndexFlatIP` é busca exata e vira o gargalo. Trocaria por HNSW ou pgvector — está citado como caminho de produção. A arquitetura não muda. |
