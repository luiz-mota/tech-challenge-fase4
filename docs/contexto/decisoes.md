# Log de decisões

Formato: data · decisão · alternativas consideradas · justificativa · status (proposta | confirmada | revertida).
Atualizar sempre que algo for decidido ou mudar. Este arquivo vira a seção de justificativas do README.

---

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
- **A busca híbrida ainda não se pagou numericamente:** o BM25 sozinho vence em recall@1 com 40% da latência. A fusão permanece porque cobre a pergunta parafraseada, que o teste atual não mede — mas isso é argumento, não medição.
- **Termo obrigatório × termo de cobertura:** *"mais raro"* é a estatística certa para o filtro obrigatório e errada para medir cobertura. Por isso *"problemas com os Correios e retirada na agência"* ainda abstém por engano.
- **O corte de DF em 150 foi ajustado nas mesmas 38 perguntas** que calibraram a mediana. Validar com perguntas novas é o próximo passo da avaliação.
- **`transformers` 5.x contra a faixa `<5` do curso:** divergência deliberada — rebaixar quebraria a camada `bertopic` já validada.
