# Log de decisões

Formato: data · decisão · alternativas consideradas · justificativa · status (proposta | confirmada | revertida).
Atualizar sempre que algo for decidido ou mudar. Este arquivo vira a seção de justificativas do README.

---

## 2026-11-17 · Duas versões do roteiro: a fala simples, o termo no rodapé do slide

- **O problema:** o roteiro de 12 slides foi escrito para se **defender** numa arguição, não para ser **falado**. "Fusão por RRF com k igual a 60" trava na boca e não informa ninguém que ainda não sabe o que é RRF. Mas apagar o termo custaria caro no sentido oposto: o enunciado avalia recuperação e fundamentação, e o avaliador precisa ver que a decisão existe.
- **Alternativas:** (a) só o roteiro técnico; (b) simplificar tudo, slide incluído; (c) reescrever o técnico em linguagem simples e perder a versão original.
- **A decisão:** duas versões coexistem. Na versão simples a **fala não tem jargão** e o **termo exato vive no rodapé técnico do slide** — `.tecnico` em `project/deck.css`, 24px monoespaçado, fixo em `bottom:64px`. Quem apresenta não precisa pronunciar; quem avalia lê.
- **Justificativa:** separa dois públicos que o mesmo slide atende ao mesmo tempo — o ouvinte, que precisa da ideia, e o leitor, que precisa da evidência de rigor. O `roteiro-video-simples.md` traz a tabela de tradução ("o que eu digo ↔ o que está escrito"), para que uma pergunta sobre o termo se responda apontando para a tela.
- **Custo assumido:** dois roteiros para manter em sincronia. Mitigado por os dois citarem os mesmos números do README em vez de repeti-los de memória — o que já tinha pegado um erro: `33.812` não existe, o corpus indexado é **39.388**.
- **Status:** confirmada. 8 slides, 5min47 (5min07 cortando o slide 5), contra o limite de 7 minutos.

## 2026-11-15 · Não usar LangChain, e dizer por quê em vez de omitir

- **O problema:** LangChain é um módulo inteiro da Fase 4 (aulas 01–05) e não aparece em nenhuma linha do projeto. A única justificativa que existia estava dentro da decisão do Gemini de 13/11, que foi **revertida** em 14/11 — ou seja, o argumento caiu junto com ela e a ausência ficou sem registro. Lacuna de rastreabilidade, não de código.
- **Alternativas consideradas:** (a) reescrever a recuperação como `Runnable` encadeado; (b) usar só o wrapper de FAISS do LangChain; (c) manter Python puro e registrar.
- **Justificativa, em ordem de peso:**
  1. **O que está em avaliação é entender a mecânica do RAG**, e a abstração esconde exatamente as peças avaliadas: a fusão RRF, a escolha do limiar, a montagem do contexto. Um `RetrievalQA` pronto responderia o requisito 2 sem demonstrar nada do requisito 2.
  2. **As peças que mais diferenciam o projeto não existem no framework.** Portão de suficiência com duas vias, termo obrigatório por IDF e guardrail de citação por comparação de string teriam de ser escritos à mão de qualquer jeito — o framework cobriria a parte fácil.
  3. **Memória.** Num processo com 0,2 GB livres, uma árvore grande de dependências custa caro, e foi a falta de memória que já tornou a geração 50× mais lenta.
- **O que se perdeu, declarado:** `.with_fallbacks()` entre provedores (aula 04) é o que o framework daria de graça e aqui não existe. Numa versão com API valeria a pena. Multi-query (aula 03.2) fica fora por outro motivo: multiplicaria chamadas num modelo que já é o gargalo, a 19,8 s por resposta.
- **Decisão irmã — sem `with_structured_output`:** a saída **é** estruturada (`{resposta, fontes, respondeu}`), mas validada por parser próprio. O método do LangChain depende de *function calling* do provedor, que um Qwen 0.5B local não tem. E o guardrail é mais forte que o schema seria: Pydantic garantiria que `fontes` é lista de inteiros, **não** que esses inteiros correspondam a reviews que foram ao prompt.
- **Status:** confirmada

## 2026-11-15 · Duas figuras geradas pelo pipeline, não desenhadas à mão

- **Alternativas:** `visualize_barchart` e `visualize_topics` do próprio BERTopic, como as aulas 04 e 06 de NLP sugerem; nenhuma figura.
- **Por que não as do BERTopic:** elas mostram os **101 temas** e os seus termos c-TF-IDF. O que o projeto quer mostrar são os **16 macro-temas**, que são a contribuição própria — e o gráfico do BERTopic não conhece esse agrupamento. Mostrar 101 barras de termo seria exibir a ferramenta, não o resultado.
- **As duas figuras, e a pergunta que cada uma responde:**
  1. `temas_problema.png` — a distribuição dos problemas, com o **denominador dentro da figura**. Gráfico de frequência sem denominador declarado é o erro que a camada de contagem inteira existe para evitar; deixá-lo fora da imagem seria reintroduzi-lo na hora da apresentação.
  2. `tema_vs_nota.png` — a taxa de atraso **real** por macro-tema contra a média do corpus. É a figura que justifica a camada existir, porque `atraso_dias` vem das datas do pedido e não do texto: é validação independente.
- **Decisões de visualização:** uma série, uma cor (colorir barra a barra sugeriria um agrupamento que não existe); barra horizontal porque são 11 categorias de nome longo; rótulo direto em vez de eixo, já que figura estática não tem tooltip; linha de referência na figura 2, sem a qual "41%" não significa nada.
- **Status:** confirmada

## 2026-11-15 · Correção: a média de atraso do corpus é 9,4%, não 10,4%

- **O defeito:** o README afirmava *"corpus: 10,4% de entregas atrasadas na média"*. O número correto sobre o corpus atual é **9,42%** (3.709 de 39.388). O 10,4% vinha da tentativa de 62 temas, antes de o funil de ingestão mudar, e sobreviveu a uma reescrita.
- **Por que importa mais do que um décimo:** é a **linha de base** contra a qual os 41,2% de "produto não recebido" são lidos. Número de referência errado contamina a leitura da tabela inteira, e é justamente o tipo de erro que ninguém confere porque parece plausível.
- **Achado ao gerar a figura** `tema_vs_nota.png`, que recalcula a base a partir do parquet em vez de repetir o texto. Figura gerada por código em vez de desenhada à mão tem esse efeito colateral: ela confere os números do texto.
- **Registro histórico mantido:** a entrada de 14/11 sobre o BERTopic continua dizendo 10,4%, porque era o que estava medido naquele momento, sobre outro corpus.
- **Status:** confirmada

## 2026-11-15 · `entrega_atrasada = False` esconde um nulo

- **O achado:** o campo é booleano e vale `False` tanto para "entregue no prazo" quanto para "sem data de entrega" — 1.834 documentos (4,7%), dos quais 619 apenas despachados, 408 indisponíveis e 381 **cancelados**. Um pedido cancelado não foi entregue no prazo; ele não foi entregue.
- **Decisão: manter e declarar, não corrigir.** O efeito é **deflacionar** a taxa de atraso, nunca inflá-la — então nenhuma conclusão do projeto fica otimista por causa disso. Separar em três estados (`atrasado` / `no prazo` / `sem data`) mudaria a assinatura do filtro, a camada de contagem e os testes do portão, a dois dias do prazo.
- **Por que registrar mesmo assim:** é exatamente o tipo de nulo que vira número errado quando ninguém olha, e o requisito 1 do enunciado pede tratamento de valores ausentes — declarar o que o booleano esconde faz parte da resposta.
- **Status:** confirmada, com limitação declarada

## 2026-11-15 · Teste da CLI dependia do locale do shell

- **O defeito:** `subprocess.run(..., text=True)` decodifica a saída do filho com o locale do sistema, que no Windows é **cp1252**. O `app.py` é chamado com `-X utf8` e escreve UTF-8, então `evidência` voltava como mojibake e a asserção falhava. O mesmo teste passava noutro shell.
- **Por que é defeito de verdade e não chateação:** teste que depende do ambiente não prova nada — ele passa onde o autor roda e falha onde o avaliador roda, e o veredito não diz nada sobre o código. Reprodutibilidade é entregável do enunciado.
- **Correção:** `encoding="utf-8"` explícito na chamada.
- **A família:** é o quarto defeito do projeto que **falha em silêncio ou falha enganosamente**, junto do cache de embeddings sem assinatura, do `aplicar_temas` não idempotente e da lista de exclusão com palavras não stemizadas.
- **Status:** confirmada

## 2026-11-15 · README reorganizado na ordem em que o enunciado avalia

- **O problema:** 1.030 linhas e 40 seções acumuladas em ordem cronológica de construção. O conteúdo estava certo e o avaliador não achava a resposta dele — "onde está o requisito 3?" não tinha resposta de uma olhada.
- **Alternativas:** README curto com as medições migradas para `docs/`; só consertar o sumário.
- **Decisão:** um documento, reordenado nos quatro requisitos do enunciado, com **resumo executivo no topo** mapeando requisito → como foi atendido → onde ler. Nenhuma medição foi removida.
- **Por que não dividir em dois:** as medições **são** a justificativa, e o enunciado pede justificativa de cada escolha. Mandá-las para outro arquivo faria o README afirmar sem mostrar.
- **O que foi acrescentado junto:** RAG × *fine-tuning* (os três sinais da aula 02), LGPD e minimização porta por porta (aula 06), exemplos de pergunta e resposta das quatro rotas, e a correção do alcance real do filtro de metadado — que o texto anterior prometia na rota de evidência e só existe na de contagem.
- **Status:** confirmada

## 2026-11-12 · Critério de inclusão é ter texto, não ter `review_comment_message`
- **Alternativas:** manter a exigência de mensagem e corrigir o README para admitir o descarte; exigir título **e** mensagem.
- **Justificativa:** o formulário do Olist tem dois campos e o cliente usa o que quer. Em **1.737 reviews o título é o único texto**, e 85 têm nota 1 — *"Não recebi o produto."* é uma delas. Exigir `review_comment_message` descartava essas reviews por causa de **onde** a pessoa digitou, não do que escreveu. A alternativa de exigir os dois campos cortaria o corpus de 38.621 para 9.829, porque a maioria dos clientes deixa o título em branco.
- **Bug que isso corrigiu:** o README já afirmava *"em 1.737 avaliações o título é o único texto, é concatenado à mensagem"*, mas o código filtrava `mensagem != ""` **antes** de concatenar — nenhuma das 1.737 era recuperada. O entregável afirmava o contrário do que o código fazia.
- **Resultado:** das 1.737, **767 sobreviveram ao piso**; o resto era `"Ótimo"`, `"bom"`, `"5"`, de uma palavra. O piso filtrou o ruído sozinho. Corpus: 38.621 → **39.388**.
- **Status:** confirmada

## 2026-11-12 · Piso de comprimento baixado de 4 para 2 palavras
- **Alternativas:** manter em 4; manter em 4 com exceção para textos com negação; trocar contagem de palavras por lista de frases vazias.
- **Justificativa:** o piso de 4 descartava 6.829 reviews (16,8% do corpus com texto), e entre elas estavam "Não chegou", "Não foi entregue", "Produto nunca chegou", "Ainda não recebi" — o tipo exato de reclamação que o enunciado pede. A nota média dos descartados com negação era 2,28, contra 4,50 dos descartados sem negação: o corte era enviesado contra a reclamação. O README anterior afirmava que o piso de 4 preservava "chegou quebrado" e "produto errado", frases de 2 palavras que ele de fato cortava — a justificativa escrita não batia com o código. Decisivo: **incluir é reversível, excluir não é** — com a coluna `n_palavras` guardada, a camada de recuperação pode despriorizar texto curto depois; reconstruir o corpus para recuperá-lo, não.
- **Custo assumido:** entram 4.809 reviews, só ~218 com negação; as demais são elogio curto sem conteúdo. A redundância do corpus sobe de 2,3% para 7,7% ("muito bom" aparece 352 vezes), e documentos curtos pontuam alto no BM25 por normalização de comprimento. Mitigação definida: colapsar textos idênticos **na apresentação**, não na ingestão — "não recebi o produto" repetido 53 vezes são 53 clientes, o que é corroboração e não duplicata.
- **Em aberto:** o piso do índice não precisa ser o do BERTopic. Modelagem de tópico sobre "muito bom" produz lixo, então a modelagem deve usar um piso maior sobre o mesmo corpus.
- **Corpus:** 33.812 → **38.621 documentos**.
- **Status:** confirmada

## 2026-11-12 · Terceira rota de limpeza, exclusiva da perna BM25
- **Alternativas:** reusar `pipeline_nlp_analitico` (o pipeline da aula 02) na perna lexical.
- **Justificativa:** reusar a rota analítica violaria a regra de nunca remover negação. Ela remove stop words, e `não`, `nem`, `sem` e `mas` são stop words do NLTK — "não recebi o produto" e "recebi o produto" virariam o mesmo saco de radicais, justamente na perna que existe para ser literal. A rota analítica pode perder a negação porque representa **frequência**, onde `não` é ruído de alta frequência; o BM25 compara **palavra com palavra** e não pode.
- **Três ajustes medidos:** (1) `não`/`nem`/`sem`/`mas` saem do conjunto removido; (2) `nunca`, `jamais`, `nada` e `nenhum` não são stop words mas o RSLP os deforma (`nunca`→`nunc`, `jamais`→`jamal`), então a negação não é stemizada; (3) `não` é stop word e `nao` sem acento não é — o cliente escreve das duas formas, e as variantes são canonizadas antes do filtro.
- **Consequência aceita:** dígitos não entram no índice lexical, porque a rota reusa `limpar_normalizar`. A evidência quantitativa ("15 dias de atraso") vive na rota de indexação, que preserva dígito; no BM25 número tende a ser ruído (preço, data, nº de pedido).
- **Gravada como coluna** `texto_bm25` no corpus, não montada na consulta: stemizar 38 mil textos não deve entrar na latência de cada pergunta.
- **Status:** confirmada

## 2026-11-15 · Fidelidade medida sem juiz de modelo, e por que não é RAGAS

- **O que o plano previa:** RAGAS para *faithfulness* e *answer relevancy*. RAGAS decompõe a resposta em afirmações e pede a um LLM juiz que diga se cada uma se sustenta no contexto.
- **Por que não dá, e não é preguiça:** não há chave de API, e um segundo modelo num processo com 0,2 GB de RAM livre é a mesma causa que já tornou a geração 50x mais lenta. Usar o Qwen 0.5B como juiz do próprio Qwen 0.5B seria **pior que não medir**: mediria a concordância de um modelo fraco consigo mesmo e chamaria isso de fidelidade.
- **O que foi medido em vez disso:** todo radical de conteúdo da resposta tem de ter origem conhecida. É ancoragem lexical, pela **mesma rota do BM25** — reusar a rota do índice é o que garante que "atrasada" na resposta e "atraso" na review virem o mesmo radical; duas normalizações mediriam contra alvo deslocado.
- **A métrica só ficou interpretável quando separou três origens**, e a primeira versão, que não separava, acusou 40% de invenção enganosamente:

      ancorado          está na review CITADA          -> sustenta a frase
      eco da pergunta   está na pergunta               -> origem conhecida; defeito de utilidade, não de fidelidade
      visto e não citado está em review que foi ao prompt -> imprecisão de CITAÇÃO, que o guardrail já trata
      inventado         em nenhum dos três             -> candidato real a alucinação

- **`taxa_invencao` é a métrica de fidelidade; `taxa` de ancoragem não é.** Ancoragem de 100% penalizaria o sinônimo e a paráfrase legítima, então não tem teto útil. Invenção tem alvo: zero, e zero é atingível.
- **Resultado final:** 7 de 8 perguntas respondidas, ancoragem mediana 75%, **invenção mediana 0%**, 3 de 7 respostas com algum radical órfão. Latência mediana 19,8 s. Custo por pergunta: R$ 0,00.
- **Os três órfãos que sobraram, nomeados um a um:** `entreg`/`form`/`inadequ` ("entregue de forma inadequada") é o modelo editorializando de verdade; `receb` é palavra ubíqua ausente das três reviews citadas daquele caso; `demor` é **artefato do stemizador** (ver a decisão do RSLP abaixo), não invenção. Declarar qual é qual é o que impede o número de ser vendido além do que ele mede.
- **Limites declarados:** não pega negação nem relação invertida ("chegou" e "não chegou" têm os mesmos radicais) e não pega paráfrase legítima. Por isso negação e conjunção foram **retiradas do escopo** da métrica: mantê-las era medir o que ela admite não medir, e o efeito prático era acusar de invenção o "não" de uma frase correta.
- ***Answer relevancy* já estava coberta de outro jeito:** o acerto por macro-tema em `calibrar.py` verifica se as reviews recuperadas pertencem ao tema que a pergunta pedia — sem juiz, porque o gabarito é humano.
- **Status:** confirmada

## 2026-11-15 · Vazamento de persona: o modelo se desculpava como se fosse o vendedor

- **Achado pela métrica de fidelidade**, não por leitura: a resposta de menor ancoragem era *"**Peço desculpas pela confusão.** Veio um produto diferente do que você pediu"*, com os órfãos `peç`, `desculp`, `confus`.
- **Não era alucinação de fato — era persona.** O modelo respondia como atendimento ao cliente em vez de relatar o que os clientes escreveram. O mecanismo é a **pergunta em primeira pessoa** ("o produto que *eu* pedi") puxando o modelo para modo de diálogo.
- **A causa no prompt:** ele dizia o que fazer e nunca dizia quem o assistente **não é**.
- **Correção:** regra 6 explícita — relata em terceira pessoa, não é o vendedor nem o atendimento, nunca se desculpa, nunca promete solução, e mesmo com pergunta na primeira pessoa responde "os clientes relatam..." — mais um exemplo few-shot com pergunta em primeira pessoa.
- **Efeito medido:** a desculpa desapareceu de todas as respostas, e as abstenções caíram de 3 em 8 para 1 em 8 (o exemplo novo também ensinou a responder).
- **Status:** confirmada

## 2026-11-15 · O exemplo few-shot estava sendo lido como evidência

- **Defeito medido:** a resposta a "o que dizem os clientes sobre a caixa amassada?" afirmou que *"o produto estava solto dentro"*. Verificação direta: a frase **não estava em nenhuma das cinco reviews recuperadas** e estava, literal, no exemplo 1 do meu prompt. O modelo usou o meu exemplo como evidência.
- **Três tentativas de guardrail, e por que as duas primeiras falharam:**
  1. **Proibir todo radical dos exemplos ausente da evidência.** Resultado: 7 de 8 respostas viraram abstenção. Causa: os exemplos contêm `produt`, `vei`, `cheg`, `receb` — as palavras mais comuns do domínio. Proibir essas é proibir falar português sobre e-commerce.
  2. **Proibir só o radical discriminativo (IDF alto), mesmo critério do termo obrigatório da busca.** Também falhou, e a medição explicou por quê de forma definitiva: `relat` tem **IDF 7,78** no corpus, mais raro que `fon` (6,51). A moldura de relato que os exemplos ensinam é, por construção, vocabulário que review nenhuma usa — logo **moldura e contaminação não são separáveis por raridade**. O confundimento é estrutural, não calibrável.
  3. **Desistir do guardrail e consertar a causa.** É o que ficou.
- **A causa real era o exemplo resumir além do que mostra:** a resposta do exemplo 3 terminava em *"como cor trocada e modelo diferente do anunciado"* — `cor` e `troc` não estavam nas reviews daquele exemplo. **Exemplo que extrapola ensina a extrapolar.** Agora a resposta de cada exemplo usa somente conteúdo das reviews do próprio exemplo, e há teste que confere isso automaticamente contra a métrica de fidelidade.
- **A detecção ficou, mas na camada de avaliação**, como `vindos_do_exemplo`: ali um falso positivo custa uma linha de diagnóstico, não uma resposta. É o lugar certo para um sinal informativo e pouco confiável — e é o oposto do guardrail de citação, que é confiável e por isso bloqueia.
- **Efeito medido:** o caso "veio um produto diferente do que eu pedi" passou de 50% de ancoragem e 50% de invenção para **100% e 0%**. Nenhum radical de exemplo aparece mais nas respostas.
- **O que isto generaliza:** guardrail só vale quando o sinal é confiável. Barrar com sinal ruidoso troca alucinação por mudez, e mudez em 7 de 8 perguntas não é segurança — é o sistema desligado.
- **Status:** confirmada

## 2026-11-15 · RSLP separa substantivo de verbo — e a busca híbrida finalmente se justificou

- **Achado ao investigar um falso positivo da métrica:** `demora` steniza para `dem` e `demorou` para `demor`. Substantivo e verbo da mesma palavra viram radicais diferentes.
- **Medição em 16 pares que um gestor usaria na pergunta contra a forma que a review usa: 5 divergem**, e o impacto é concreto:

      pergunta "cobrança"  -> cobranc  em     0 docs   |  review "cobrar"     -> cobr    em 200 docs
      pergunta "devolução" -> devoluca em    14 docs   |  review "devolver"   -> devolv  em 371 docs
      pergunta "danos"     -> dan      em    33 docs   |  review "danificado" -> danific em 106 docs
      pergunta "demora"    -> dem      em   355 docs   |  review "demorou"    -> demor   em 583 docs

- **A hipótese de que isso viraria falso abstimento foi testada e REFUTADA:** um radical ausente do corpus tem IDF 0, que é o mínimo, então ele nunca é escolhido como termo obrigatório — e há guarda explícita para o caso de o termo escolhido não existir.
- **O dano real é perda de recall LEXICAL, e a perna vetorial compensa.** "problemas com a devolução do produto" teve a perna lexical limitada a 14 candidatos em vez dos 371 que dizem "devolver" — e o top-1 veio correto ("solicito a devolucao do produto"), mediana 0,779, aprovado pela via semântica.
- **Isto fecha uma pendência que estava aberta e honesta.** O registro anterior dizia que a busca híbrida *"ainda não se pagou numericamente"*, porque o BM25 sozinho vence em recall@1 com 40% da latência, e que a defesa da fusão era argumento e não medição. **Agora é medição:** existe uma classe de pergunta — a que usa a forma substantiva — em que a perna lexical degrada por artefato de stemming e só a vetorial salva. A fusão não é redundância.
- **"problemas de cobrança" é o caso que ainda degrada**, e o sistema admite: passou pela via de vocabulário na palavra errada (`problem`, 1.010 documentos) e trouxe "Problemas com os Correios". Foi marcado **`evidência FRACA`** pelo grau de suficiência novo — o defeito não foi corrigido, foi tornado visível.
- **Trocar o stemizador foi descartado** a dois dias do prazo: implicaria reconstruir o índice lexical, a camada de temas e toda a calibração, com risco de invalidar o mapa de 101 temas feito à mão. A mitigação já existe na arquitetura e o residual é sinalizado.
- **Status:** confirmada, com limitação declarada

## 2026-11-15 · Dois defeitos de reprodutibilidade, achados ao reconstruir

- **`aplicar_temas` não era idempotente.** Ele gravava em `corpus_temas.parquet` e também o lia — então na segunda execução os 10.212 documentos atribuídos por centroide entravam no cálculo dos próprios centroides, que se deslocavam. Rodar duas vezes dava resultado diferente de rodar uma, e o pipeline que o README diz ser reprodutível não era. **Correção sem retreinar nada:** a coluna `origem` recupera o conjunto de treino do próprio arquivo de saída, e um espelho em `temas_treino.parquet` torna a dependência explícita. Verificado: segunda execução devolve 27.941 / 10.212 / 1.235 / 13.166, idêntico.
- **A lista de exclusão da métrica falhava em silêncio.** `MOLDURA_RESPOSTA` é comparada contra radicais do RSLP, e quatro entradas estavam escritas por extenso — `caso`, `comprador`, `principal`, `porque`, quando os radicais são `cas`, `compr`, `princip`, `porqu`. Lista de exclusão que não casa **não quebra: só deixa de excluir**. Há teste que confere cada entrada contra o stemizador.
- **O que os dois têm em comum:** nenhum dos dois falha de forma visível. Um devolve número plausível com dado velho, o outro deixa de filtrar sem avisar. São o mesmo tipo de defeito do cache de embeddings sem assinatura, e é por isso que os três viraram teste ou verificação automática.
- **Status:** confirmada

## 2026-11-15 · Mascaramento de PII na ingestão, e por que o detector da aula precisou de uma guarda

- **Regra 9 do projeto não estava cumprida.** As rotas de limpeza removiam e-mail, mas o campo `texto` — que é o que vai para o embedding, para o prompt do LLM e para a tela do `:fontes` — não passava por nenhum mascaramento. Dado pessoal não mascarado na ingestão vaza nos três lugares.
- **Base:** padrões e validadores da aula 06 de IA Generativa, sem alteração. O que ela ensina e que decide a arquitetura: a **regex propõe o candidato e a aritmética decide**. Dígito verificador de CPF e algoritmo de Luhn derrubam o falso positivo que a regex sozinha aceitaria — `123.456.789-00` tem a forma de um CPF e não é um.
- **Medição antes de implementar**, com os padrões da aula sem mudança, nas 39.388 reviews:

      documentos com PII validada ......  10  (0,025%)
        PII real ......................   2
        falso positivo ................   8

- **Os 8 falsos positivos têm todos a mesma causa:** em review de e-commerce, oito dígitos seguidos quase nunca são telefone ou CEP. São **número de pedido** ("Pedido 02-671663252", "Pedido: 01-68698482") e **data** sem separador ("01092018"). O padrão de telefone casa `\d{4}[\s.-]?\d{4}` e o de CEP casa `\d{5}-?\d{3}`: os dois têm exatamente a forma de um código de pedido.
- **Acrescentado: a guarda de número de pedido.** Dígitos precedidos de marcador (`pedido`, `protocolo`, `rastreio`, `nota fiscal`) não são tratados como PII. Isso não é conveniência — **número de pedido não é dado pessoal**, é referência de transação, e é evidência útil numa reclamação de "pedido não entregue". Mascará-lo destruiria informação legítima sem proteger ninguém. A guarda eliminou 2 dos 8.
- **Por que mascarar mesmo com os falsos positivos que sobraram:** a assimetria resolve. Um falso positivo custa alguns asteriscos em 0,02% do corpus; um falso negativo publica o telefone de uma pessoa real — e há um no corpus: *"Faltou item descrito Me ligue 34 99978 4951"*. Com esse desequilíbrio, o detector deve errar para o lado de mascarar. Não foram caçados um a um.
- **Formato da máscara:** `*********51`, preservando os dois últimos. Quem escreveu reconhece o próprio dado; o avaliador vê que houve mascaramento e qual era o tamanho. `[REMOVIDO]` apagaria a evidência de que a proteção agiu.
- **Limitação declarada:** **nome de pessoa não é detectado.** Não há regex para nome, e NER em português seria um quarto modelo num processo com 0,2 GB de RAM livre. O que protege parcialmente é a origem: zero CPF e zero e-mail validados em 39.388 reviews é consistente com o Olist já ter anonimizado a base, não com sorte.
- **Resultado após reconstruir o corpus:** 8 documentos mascarados, 39.388 mantidos. 17 testes fixam os casos medidos.
- **Status:** confirmada

## 2026-11-15 · Política de entrada: o primeiro dos três casos de recusa

- **A regra 3 pede três casos distintos** e só dois existiam. Faltava o primeiro: recusar **antes de buscar**.
- **A aula 06 separa duas políticas:** bloquear PII na entrada e mascarar na saída. A saída passou a estar coberta na ingestão (nenhuma PII chega ao corpus), então o que faltava era a entrada — a pergunta do gestor.
- **Bloquear e não mascarar a pergunta.** Um CPF na pergunta não é acidente de formatação: é a intenção de consultar uma pessoa. Este sistema responde sobre padrões e depoimentos do conjunto, não sobre indivíduos. Mascarar transformaria a pergunta em outra, e responder a outra pergunta é pior que recusar esta.
- **Limite de 2.000 caracteres**, da política da aula. Pergunta longa não é pergunta: é tentativa de encher o contexto para empurrar instrução junto — e o orçamento de tokens do modelo de 0,5B torna o limite concreto, não teórico.
- **A guarda de número de pedido vale aqui também:** *"quantos clientes reclamaram do pedido 02-671663252?"* é pergunta legítima e não bloqueia. Há teste para isso.
- **Status:** confirmada

## 2026-11-15 · Portão passa a ter grau, não só veredito — o terceiro caso de recusa

- **O que faltava da regra 3:** o terceiro caso é *"responde, mas com `suficiencia` baixa"*. O portão era binário e tratava igual passar raspando e passar com folga.
- **Regra:** `alta` quando passou pela via semântica **e** tem ao menos 4 de 5 documentos citáveis; `baixa` quando passou mas (a) só pelo vocabulário — o corpus fala do assunto e nada se parece muito com a pergunta — ou (b) tem 2 ou 3 citáveis, caso em que a afirmação fica sustentada pela minoria do que o modelo leu; `nenhuma` quando o portão fechou.
- **O caso de "reclamações sobre os Correios" é exatamente o do meio:** mediana 0,553, termo central em 1.031 documentos. Calar seria errado, porque a evidência existe. Afirmar no mesmo tom de quando há folga também. A resposta sai com ressalva visível.
- **Por que 4 de 5:** em 4 de 5, a maioria do que foi lido sustenta a afirmação. Em 3 de 5, dois dos documentos que foram ao prompt estão abaixo do limiar de documento.
- **O grau atravessa o sistema inteiro:** `Resultado.suficiencia` → `Resposta.suficiencia` → ressalva impressa na CLI. Não é um campo de log, é parte da resposta.
- **Status:** confirmada

## 2026-11-15 · Cache de embeddings ganha assinatura — era uma armadilha silenciosa

- **Defeito encontrado ao reconstruir o índice** depois do mascaramento: o cache de embeddings checava apenas `len(emb) == len(df)`. O mascaramento alterou 8 documentos sem mudar a contagem de linhas, e o índice **reusou o vetor antigo dos oito sem avisar**. O log dizia "embeddings do cache" e estava tecnicamente correto.
- **Por que isso é pior que um erro:** um erro quebra e é consertado. Este devolvia resultado plausível com dado velho — e a mesma falha aconteceria ao mudar qualquer regra de limpeza.
- **Correção:** assinatura SHA-256 do texto de todos os documentos **mais o nome do modelo de embedding**, gravada ao lado do `.npy`. O nome do modelo entra porque trocar de modelo com o mesmo corpus é a outra metade da mesma falha, e a dimensão não muda necessariamente.
- **Separador nulo entre documentos** no hash, para que a concatenação de dois textos diferentes não colida com a de outros dois.
- **O log passou a dizer o motivo** do recálculo em vez de só anunciar o resultado: *"corpus ou modelo mudaram — cache invalidado"*.
- **Status:** confirmada

## 2026-11-14 · Interface: CLI, por restrição de memória medida

- **Decisão do aluno**, entre CLI, Streamlit, notebook e FastAPI.
- **Razão que decidiu:** a mesma que já havia derrubado o modelo de 1,5B. RAM total 7,9 GB, **livre 0,2 GB**, commit em uso 26,3 GB. O processo já carrega o índice FAISS (39.388 × 384), o BM25 serializado, o corpus em pandas, o MiniLM e o Qwen. Streamlit acrescenta um servidor e um **segundo processo Python** ao mesmo orçamento — e foi a falta de memória, não a CPU nem a biblioteca, que tornou a geração 50× mais lenta na medição deste dia. Escolher Streamlit seria repetir de propósito o erro que acabou de ser diagnosticado.
- **Ganho que não era o motivo, mas conta:** a CLI imprime a **decisão**, não só a resposta — qual rota o roteador escolheu, por qual das duas vias o portão passou, e a causa da abstenção quando ela acontece (`portao` antes de chamar o modelo, em ~0,6 s; `modelo` depois de ver as evidências). Caixa de chat esconde isso, e é justamente isso que precisa ser auditado.
- **`:fontes` é a rastreabilidade deixando de ser promessa:** o comando procura no mapa de citação cada `review_id` que a resposta citou e imprime o texto original com nota, categoria, UF e se a entrega atrasou. Quem avalia confere frase por frase se a resposta se sustenta na evidência. Sem esse comando, a citação é uma string hexadecimal que ninguém verifica.
- **`--sem-llm` existe pela mesma restrição:** sobe em 0,4 s contra 18 s e responde as rotas de contagem e de série temporal, que nunca passaram pelo LLM. E **recusa no roteador**, em milissegundos, uma pergunta de evidência — carregar o corpus inteiro para então levantar `RuntimeError` gastaria 18 s de quem rodou. Há teste para isso.
- **Descartado:** `notebook` como interface principal (exige Jupyter no ambiente do avaliador e mistura execução com narrativa); `FastAPI` (acrescenta servidor sem nenhum cliente que o consuma — seria arquitetura para ninguém).
- **O notebook não foi descartado como peça:** ele continua sendo onde a análise exploratória mora. O que ele não é, é a porta de entrada do sistema.
- **Barras de progresso do `transformers` silenciadas por variável de ambiente** antes do import, não por chamada de função: a biblioteca lê isso no momento em que é importada.
- **Status:** confirmada

## 2026-11-14 · Portão de suficiência ganha uma segunda via: cobertura de vocabulário
- **Falha que motivou:** "reclamações sobre os Correios" recuperava *"Problemas com os Correios"* e *"NÃO CHEGOU POR PROBLEMAS NOS CORREIOS"* — recuperação correta — e era **barrada** pelo portão, com mediana 0,553. A falha não é aleatória: pergunta curta recupera documento curto, e documento curto tem cosseno menor. O limiar de 0,689 derruba sistematicamente pergunta legítima de formulação curta.
- **Sinal acrescentado:** quantos documentos do corpus contêm o termo central da pergunta (DF). Não é similaridade, é **cobertura de vocabulário** — se a palavra central quase não existe no corpus, a pergunta é de outro assunto, e nenhuma similaridade conserta isso. É o que a palavra "Brasil" numa review de toalha falsifica no cosseno e não falsifica aqui.
- **Medição nas 38 perguntas:** respondíveis com DF mediano 281 (máx. 2.515); fora do corpus com DF mediano 12 e **máximo 105**. Corte em 150, com 45 de margem sobre o pior caso inválido.
- **Regra final:** passa se `mediana >= 0,689` **OU** `DF >= 150`. As duas vias são independentes e cada uma cobre a cegueira da outra, pelo mesmo princípio das duas pernas da busca. O filtro de anedota (mínimo 2 documentos citáveis) vale para as duas. A resposta registra **por qual via passou**, o que mantém a decisão auditável.
- **Resultado:** 37/38, sem regressão — nenhuma pergunta fora do corpus passou pela segunda via. O caso dos Correios passou com DF de 1.031.
- **Sozinho, o sinal novo é PIOR:** 92% contra 97% da mediana. Ele serve como via alternativa, não como substituto.
- **Limitação do método, declarada:** o corte de 150 foi escolhido olhando as mesmas 38 perguntas que já tinham servido para calibrar a mediana. A separação entre calibração e validação **se perdeu nesta rodada**. O 97% aqui não é medição independente como o 100% anterior era. Validar com perguntas novas é o próximo passo da avaliação, não um feito.
- **Falha remanescente, com mecanismo identificado:** "problemas com os Correios e retirada na agência" continua barrada (mediana 0,576, DF 54). O termo escolhido foi `agênc` (54 docs) e não `correi` (1.031), porque o termo obrigatório é o de **maior IDF**, isto é, o mais raro — e o mais raro aqui é a palavra periférica, não o assunto. *"Mais raro" é a estatística certa para o filtro obrigatório e errada para medir cobertura.* A correção óbvia (usar o termo de maior DF) deixaria "presidente do **Brasil**" passar. Fica em aberto: separar a escolha do termo obrigatório da escolha do termo de cobertura.
- **Status:** confirmada, com pendência registrada

## 2026-11-14 · Citação por número da evidência, não pelo review_id
- **O material já fazia assim** e eu havia divergido sem motivo: a aula 02 de IA Generativa numera as evidências e pede `Fontes: [n]`.
- **Problema medido:** um `review_id` tem 32 caracteres hexadecimais e custa ~20 tokens. Com cinco citações, as fontes consumiam ~100 dos 140 tokens de orçamento e o JSON era **truncado antes de fechar**. O parser falhava e uma resposta correta virava abstenção falsa — perda por orçamento, não por conteúdo.
- **Correção:** evidências entram como `<review n="1" nota="...">` e o modelo cita `[1, 2]`, que custa 2 tokens. O código resolve número → `review_id`.
- **A rastreabilidade não se perde:** o id que vai para a resposta final continua sendo o real, resolvido pelo código. O modelo nunca precisou saber o id — só precisa apontar qual das cinco evidências usou. Menos trabalho para o modelo, mesma auditabilidade.
- **O guardrail ficou mais forte:** agora rejeita número fora da faixa 1..N **e** id inexistente, e registra qual foi o problema. O que ele perdoa é a **formatação** da citação; o que nunca perdoa é citar review que a busca não recuperou.
- **Status:** confirmada

## 2026-11-14 · Few-shot com exemplo positivo no prompt de geração
- **Lacuna própria:** a aula 03 ensina a progressão zero-shot → instrução fechada → few-shot, e o prompt inicial tinha **só exemplo de recusa**. O modelo nunca viu como é uma resposta boa.
- **Antes:** `"resposta": "Ruim"` / `"Não recebi"`. **Depois:** *"Os clientes afirmam que o produto não foi entregue"*.
- **Status:** confirmada

## 2026-11-14 · Comparação de molduras de pergunta ignora acento
- **Defeito medido:** "reclamações sobre os Correios" e "reclamacoes sobre os Correios" produziam recuperação **diferente**, porque só a primeira casava com a lista de moldura. Quem digita pergunta omite acento, e o sistema não pode depender disso.
- **Correção:** a lista sem acento é **derivada** da original, não escrita à mão — lista duplicada sai de sincronia na primeira manutenção.
- **Status:** confirmada

## 2026-11-14 · Modelo de geração: 0.5B por restrição de memória, não por preferência
- **Medição:** 721 s e 580 s por resposta com o Qwen 1.5B. A causa não era versão de biblioteca nem número de threads — era **memória**: RAM total 7,9 GB, **livre 0,2 GB**, commit em uso 26,3 GB, CPU i5-8265U (4 núcleos, 1,6 GHz). 1,5B em float32 são ~6 GB só de pesos, ao lado do FAISS, do corpus e do MiniLM. O Windows passou a paginar em disco e cada token virou leitura de disco.
- **Descartado no caminho:** `float16` em CPU, que é mais lento que `float32` por falta de kernel nativo; e rebaixar `transformers` para a faixa `<5` do curso, que quebraria o `bertopic` já validado (divergência registrada, não resolvida).
- **Correção:** `Qwen2.5-0.5B-Instruct`, que o próprio `setup/baixar_modelos.py` do curso lista como *"fallback (velocidade/VRAM)"*. Resultado: **13 a 18 s** por resposta, ~50× mais rápido. Mais `accelerate` instalado (o curso exige e faltava), 8 threads em vez de 4, e `max_new_tokens` de 220 para 140.
- **Por que isso não afeta a correção:** o guardrail confere `review_id` por comparação de string, a contagem vem do `pandas`, o portão vem do cosseno. O modelo só escreve a frase final. Modelo menor escreve pior e **não cita falso** — por construção, não por confiança.
- **O modelo maior fica selecionável:** `MODELO_GERACAO=Qwen/Qwen2.5-1.5B-Instruct`, sem editar código.
- **Status:** confirmada

## 2026-11-14 · LLM de geração: Qwen2.5-1.5B-Instruct local — reversão da decisão de 13/11
- **Revoga:** a decisão de 13/11 (Google AI Studio como principal, Ollama como fallback).
- **Alternativas:** Gemini Flash via API (tier gratuito); Groq; Ollama local.
- **Justificativa:** conferência contra o material do curso mostrou que as aulas 02 e 03 de IA Generativa fazem a geração **100% local com `transformers` e Qwen2.5-1.5B-Instruct**, e que `transformers` e `torch` **já estavam instalados** como dependência do BERTopic. Pesquisa sobre o tier gratuito do Gemini devolveu números contraditórios entre fontes (de 20 a 1.500 requisições/dia), nenhuma oficial, e mudança de política em abril/2026 restringindo modelos e introduzindo faturamento pré-pago. Decisivo: **o avaliador precisa executar o projeto**, e com API ele precisaria de chave própria. Ollama cai junto porque exigiria instalação fora do `requirements.txt`.
- **Custo assumido:** qualidade de redação menor que um modelo de API. Aceitável porque o guardrail valida `review_id` por código: **modelo pequeno escreve pior, não cita falso**. A correção da resposta não depende da qualidade do modelo.
- **Status:** confirmada

## 2026-11-14 · Portão de suficiência: mediana do cosseno do top-k em 0,689
- **Alternativas medidas (22 perguntas, 12 respondíveis e 10 fora do corpus):** cosseno máximo (88%), média do top-k (95%), contagem acima de 0,60 (92%), sobreposição entre as pernas (82%).
- **Justificativa:** o cosseno **máximo não separa** pergunta respondível de pergunta fora do corpus. "quem é o presidente do Brasil?" casou com "Toalha do maior campeão do Brasil" a 0,752, acima da pergunta legítima sobre Correios (0,630) — uma palavra em comum basta. A mediana exige que o **conjunto** recuperado seja bom, e coincidência lexical não acontece cinco vezes.
- **Hipótese descartada:** a sobreposição entre BM25 e vetorial parecia o sinal natural (dois métodos independentes concordando seria evidência) e foi **o pior dos cinco**. "qual o telefone do atendimento?" produziu sobreposição 5 estando fora do corpus; "quem ainda não recebeu" produziu 0 estando dentro. As pernas concordam sobre ruído quando o ruído tem palavras frequentes.
- **Validação fora da amostra:** 16 perguntas novas, que não participaram da escolha do corte. Calibração 95% (11/12 válidas aceitas, 10/10 inválidas recusadas), validação 100% (10/10 e 6/6). Não há sinal de sobreajuste ao conjunto de escolha. Ressalva: 16 perguntas é amostra pequena.
- **Status:** confirmada

## 2026-11-14 · Termo obrigatório por IDF na perna lexical
- **Não existe no material do curso** — é acréscimo nosso, motivado por falha medida.
- **Falha:** "reclamações sobre os Correios" devolvia "Sem reclamações", "Nada a reclamar". Duas causas somadas: a palavra `reclamações` da pergunta virava o alvo, e documento de 2 palavras pontua alto no BM25 por normalização de comprimento — o custo do piso 2, materializado.
- **Correção:** (1) remover a moldura da pergunta (`reclamações`, `clientes`, `dizem`, `sobre`, `quais`) apenas da perna lexical, porque a vetorial precisa da frase inteira ("quem não recebeu" != "quem recebeu"); (2) exigir no documento o termo de **maior IDF** da consulta. O IDF vem do próprio índice: é raridade medida nos 39.388 documentos, não julgamento sobre qual palavra importa. Quando nenhum termo passa o piso de IDF, o filtro se desliga sozinho.
- **Resultado:** passou a devolver "Problemas com os Correios" e "NÃO CHEGOU POR PROBLEMAS NOS CORREIOS".
- **Status:** confirmada

## 2026-11-14 · 101 temas agrupados em 16 macro-temas, por leitura e não por algoritmo
- **Alternativa descartada:** agrupamento hierárquico automático sobre os centroides. Testado: 25.000 dos 28.827 documentos caíram em 3 grupos, o maior sendo "tudo que é elogio" (12.618 docs, nota 4,73) e o segundo "tudo que é problema de entrega". Os grupos pequenos viraram `relógio`, `mochila`, `cortina`, `cadeira` — categoria de produto.
- **O padrão por trás das duas falhas:** o agrupamento automático encontra os eixos que o projeto **já tem de graça** — polaridade (está em `review_score`) e categoria de produto (está em `categoria`, vinda dos metadados). O eixo que falta, **tipo de problema**, não sai de proximidade vetorial.
- **Critério de leitura, nesta ordem:** `delta_atraso_pp` (metadado, não opinião) separa "não recebeu" de "recebeu com problema"; a nota separa reclamação de elogio; as palavras dizem qual problema é.
- **Validação:** os macro-temas de entrega e pós-venda ficaram com 15 a 41% de entrega atrasada, os de produto com 3 a 6% — com nota igualmente ruim. Se a leitura estivesse errada, essa coluna estaria misturada.
- **Macro-tema `misto`:** 7 temas cujas reviews tratam de vários problemas na mesma frase (o tema 35 reúne "nada da cadeira há 30 dias", "pedi lilás e veio roxo" e "o braço quebrou"). O que uniu essas reviews foi o **produto** ser cadeira, não o problema ser o mesmo. Ficam citáveis na busca e **fora da contagem** de "principal problema". Tirar os 7 afinou as duas pontas: `entrega_atrasada` foi de 30,8% para 38,8% de atraso e `qualidade_abaixo` de 3,6% para 2,7%.
- **Revisão:** 13 temas ambíguos foram lidos com o usuário nas reviews de exemplo — 7 viraram `misto`, 4 confirmados, 2 corrigidos (o 100 era "atraso" e é elogio de entrega; o 61, nota fiscal em nome errado, ganhou macro próprio porque trava garantia e troca).
- **Status:** confirmada

## 2026-11-14 · Temas aplicados ao corpus inteiro, treinados só no piso 6
- **Justificativa:** treinar e aplicar não precisam ter a mesma cobertura. O treino usa piso 6 porque tópico sobre "muito bom" produz lixo; a aplicação usa os 39.388 porque, depois de treinado, medir distância de um documento a um centroide é perfeitamente definido.
- **Defeito que corrigiu:** a avaliação marcava "ERROU" em perguntas que a busca acertava — pergunta curta recupera documento curto, documento curto não tinha tema, e o gabarito não conseguia pontuar. Três das cinco falhas eram isso: defeito da medição, não da busca.
- **Resultado:** 10.211 documentos novos classificados, similaridade mediana **0,824**. Sem tema caiu para 1.236 (3,1%). Mesmo piso de 0,55 da realocação de outliers, pela mesma razão.
- **Status:** confirmada

## 2026-11-14 · Duas camadas de abstenção, não uma
- **O material faz diferente:** a aula 02 de IA Generativa põe a recusa **apenas no prompt** do LLM ("Se as evidências não contiverem a resposta, diga exatamente..."), sem portão na recuperação.
- **Por que divergir:** o LLM não pode recusar o que ninguém lhe disse que é fraco, e um modelo de 1,5B obedece mal instrução de recusa. As duas camadas são independentes: o portão não depende de obediência e a instrução pega o caso em que o portão passa evidência medianamente parecida mas que não responde.
- **Terceira camada, de fato:** o guardrail de citação. Resposta afirmativa sem nenhum `review_id` válido entre as evidências é convertida em abstenção por código.
- **Status:** confirmada

## 2026-11-14 · Defesa contra prompt injection no prompt de geração
- **Justificativa:** a aula 03 demonstra injection num e-mail; aqui o "dado" são as reviews, escritas por qualquer cliente. Varredura dos 39.388 documentos com oito padrões ("ignore as instruções", "desconsidere", "aja como", "system prompt", "classifique como"...) achou **zero ataques reais** — um único falso positivo, "aguardando novas instruções para a devolução".
- **Decisão:** a defesa entra mesmo assim (delimitadores `<review id=...>`, declaração de que ali é dado e não instrução, e exemplo de ataque no prompt), porque o custo é zero e o corpus pode crescer. A própria aula avisa que defesa por prompt reduz mas não elimina.
- **Status:** confirmada

## 2026-11-14 · Em aberto: a busca híbrida ainda não se justificou por número
- **Medição (12 perguntas, gabarito por macro-tema):** vetorial pura `recall@1` 6/12, `@5` 9/12, 24 ms. BM25 pura 8/12 e 10/12, 40 ms. Híbrida 7/12 e 10/12, 102 ms. **O BM25 sozinho ganha no top-1 e empata no resto, com 40% da latência.**
- **Três leituras possíveis, ainda não distinguidas:** (a) 12 perguntas é amostra pequena e a diferença é ruído; (b) o gabarito é generoso — basta um documento do macro-tema certo no top-5; (c) a híbrida realmente não ajuda neste corpus. O argumento a favor dela não é a média, é **não falhar feio em nenhum tipo de pergunta**, e com 12 perguntas isso não aparece na média.
- **Decisão:** manter a híbrida (o sistema funciona com ela) e registrar a pendência. Se houver tempo, medir perna por perna e decidir com dado.
- **Status:** pendência registrada

## 2026-11-14 · BERTopic viabilizado; parâmetros calibrados por medição
- **Risco eliminado:** o `CLAUDE.md` alertava que o notebook da aula 6 avisa que LDA/BERTopic não instalam no ambiente do curso. Testado em 13/11: instala e roda em Python 3.14 — `numba 0.68.1` e `hdbscan 0.8.44` têm wheel para cp314. O plano B (KMeans + TF-IDF) **não é necessário**. Embeddings do corpus de modelagem: 6,1 min em CPU, 4 threads, sem GPU.
- **Tentativa 1 descartada (`min_topic_size=150`, UMAP default):** 4 temas, 0,6% de outliers, e os dois maiores eram "nota 2,28" e "nota 4,73". O modelo separou **polaridade, não assunto** — e polaridade já é grátis em `review_score`. Causa: o eixo de maior variância num corpus de review é gostei × não gostei, e o UMAP com vizinhança global achata o resto nele. Sinal de diagnóstico: 0,6% de outlier é anormalmente baixo; agrupamento saudável deixa 10–20% de fora.
- **Tentativa 2 (`min_topic_size=50`, `n_neighbors=10`):** 62 temas e os temas passaram a discriminar assunto. **Validação independente do texto:** temas de entrega com 31–56% de entrega atrasada contra 10,4% do corpus, e temas de produto ("cor errada", "caixa amassada", "foto diferente") em 3–4% com a **mesma** nota ruim. A nota não distingue esses grupos; o tema distingue. Resto: 42,9% de outliers, alto demais.
- **Configuração final (`min_topic_size=30` + realocação com piso):** 101 temas, outliers de 46,4% para **3,1%**. A similaridade mediana dos 12.485 realocados foi **0,763**, bem acima do corte de 0,55 — não foi encaixe forçado, eram reviews que pertenciam a um tema e o HDBSCAN deixou fora por ser conservador.
- **Alternativa descartada:** `BERTopic.reduce_outliers`, que realoca **todos**. Dá cobertura de 100% ao preço de afirmar tema onde não há — e a contagem por tema é justamente o número que a resposta vai citar. Mesmo raciocínio do limiar de suficiência, aplicado à camada de tema: melhor 886 reviews sem tema que 886 contadas no tema errado.
- **Piso de palavras da modelagem: 6**, contra 2 do índice — resolve o item que ficou em aberto em 12/11. Cobre 73,2% do corpus (28.827 docs). Custo: "não chegou" sai da **formação** dos temas, mas continua no índice e citável. Aceitável porque o tema de atraso se forma com as reviews longas — confirmado pelos temas 8 e 15.
- **Stop words de domínio na nomeação:** (1) a Olist anonimizou vendedores com nomes de Game of Thrones, e `lannister` aparece em 3,08% do corpus — como nome próprio frequente ele vencia o c-TF-IDF e virava rótulo ("lannister, loja, contato" é vendedor, não assunto); (2) `produto`, `comprei`, `recomendo`, `loja`, `recebi` descrevem **o ato de avaliar**, não o assunto avaliado. Ambos saem apenas da **nomeação** — as reviews permanecem no índice e citáveis.
- **Em aberto:** 101 temas é granular demais para uma resposta. Vários são o mesmo assunto (12, 14, 24 e 6 são todos "chegou antes do prazo"). Falta uma camada de agrupamento em macro-temas antes de responder "principais problemas".
- **Status:** confirmada

## 2026-11-13 · Modelo de embedding: `paraphrase-multilingual-MiniLM-L12-v2`
- **Alternativas:** `multilingual-e5-base` (melhor em benchmark PT, 1,1 GB); BERTimbau (treinado só em PT, mas não é modelo de similaridade); embedding via API paga.
- **Justificativa:** é o modelo do material do curso, roda local e gratuito, 470 MB. A escolha prioriza **reprodutibilidade pelo avaliador** e alinhamento com o que foi estudado, não o topo do benchmark. O e5-base fica citado como caminho de melhoria.
- **Custo assumido:** qualidade de recuperação em português inferior ao e5-base. Mitigação: a perna BM25 da busca híbrida cobre termo exato, que é justamente onde o MiniLM dilui.
- **Status:** confirmada

## 2026-11-13 · LLM de geração: Google AI Studio (Gemini Flash) com Ollama como fallback
- **Alternativas:** Groq (tier gratuito, limite por minuto); só Ollama local; provedor pago.
- **Justificativa:** não há assinatura disponível, então a opção precisa ser gratuita. Dois motivos pelo arranjo com fallback: (1) o avaliador consegue rodar o projeto **sem chave nenhuma**, caindo no Ollama local; (2) materializa a proteção "plano B" da aula 4 de LangChain — o fallback deixa de ser teoria e vira parte da arquitetura, junto com cache e autocorreção.
- **Custo assumido:** o caminho local tem qualidade de geração menor. Aceitável porque o guardrail de citação valida `review_id` por código, independente do modelo — um modelo pior gera texto pior, mas não gera citação falsa.
- **Em aberto:** nada instalado nem configurado ainda (13/11). Só o desenho está definido.
- **Status:** confirmada

## 2026-11-12 · Uma review = um documento, sem chunking
- **Alternativas:** chunking fixo/recursivo.
- **Justificativa:** reviews são curtas; o texto inteiro já é a unidade semântica. Cortar só gera ruído (IA Gen aula 2).
- **Status:** proposta

## 2026-11-12 · Reviews sem texto fora do índice, dentro da contagem
- **Alternativas:** descartar; indexar só a nota.
- **Justificativa:** sem texto não há evidência citável, mas a nota continua sendo sinal estatístico.
- **Status:** proposta

## 2026-11-12 · Duas versões de limpeza, sem remover negação
- **Alternativas:** limpeza única.
- **Justificativa:** embeddings e LLM perdem informação com limpeza pesada; BM25 e tópicos ganham com ela. A negação inverte o sentido de reclamações (NLP aula 2).
- **Status:** proposta

## 2026-11-12 · Busca híbrida BM25 + vetorial com RRF k=60
- **Alternativas:** só vetorial; só BM25.
- **Justificativa:** BM25 cobre termos exatos (transportadora, "nota fiscal"), vetorial cobre paráfrase. RRF combina sem normalizar escores (IA Gen aula 2).
- **Status:** proposta

## 2026-11-12 · FAISS local, sem serviço gerenciado
- **Alternativas:** Azure AI Search, Bedrock KB, Vertex RAG Engine, pgvector.
- **Justificativa:** volume pequeno, custo zero, reprodutível por qualquer avaliador. Gerenciado fica citado como caminho de produção.
- **Status:** proposta

## 2026-11-12 · Camada de agregação por tema para perguntas de frequência
- **Alternativas:** RAG puro.
- **Justificativa:** RAG devolve as reviews mais parecidas, não as mais frequentes. Contar exige agregação (BERTopic ou classificador) antes do RAG.
- **Status:** proposta

## 2026-11-12 · Workflow fixo, não agente
- **Alternativas:** agente ReAct com tools.
- **Justificativa:** caminho previsível e codificável; mais fácil de testar, justificar e demonstrar (IA Gen aula 4).
- **Status:** proposta

## 2026-11-12 · Saída estruturada com review_id obrigatório + guardrail de citação
- **Alternativas:** resposta em texto livre com citações no corpo.
- **Justificativa:** citação validável por código atende diretamente os critérios de rastreabilidade e "evitar informação sem suporte".
- **Status:** proposta

## Pendentes de decisão
- Re-rank com cross-encoder: entra ou fica como melhoria futura.

## Pendências técnicas abertas (registradas, não resolvidas)
- **Filtro de metadado não chega à rota de evidência.** O roteador extrai UF e "só atrasadas" da pergunta, e `Agregador._filtrar` aplica o recorte — mas `BuscaHibrida.buscar()` não tem parâmetro de filtro, e `pipeline.perguntar()` descarta `rota.estado` e `rota.so_atrasadas` ao chamar o gerador. É o item de maior retorno no critério de recuperação, e o README foi corrigido em 15/11 para declarar o alcance real em vez de prometer o filtro completo.
- **Persona do prompt sem senioridade nem limite de atuação.** A aula 03 pede área + senioridade + objetivo + limites; a atual tem objetivo e limite.
- ~~A busca híbrida ainda não se pagou numericamente.~~ **Fechada em 15/11:** a divergência do RSLP entre substantivo e verbo degrada a perna lexical numa classe inteira de pergunta, e a perna vetorial salva. Ver a decisão do RSLP. O BM25 sozinho continua vencendo em recall@1 no conjunto de 12 perguntas, que não contém essa classe.
- **Termo obrigatório × termo de cobertura:** *"mais raro"* é a estatística certa para o filtro obrigatório e errada para medir cobertura. Por isso *"problemas com os Correios e retirada na agência"* ainda abstém por engano.
- **O corte de DF em 150 foi ajustado nas mesmas 38 perguntas** que calibraram a mediana. Validar com perguntas novas é o próximo passo da avaliação.
- **`transformers` 5.x contra a faixa `<5` do curso:** divergência deliberada — rebaixar quebraria a camada `bertopic` já validada.
