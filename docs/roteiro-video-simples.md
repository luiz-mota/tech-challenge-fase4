# Roteiro do vídeo — versão simples

Deck pronto para apresentar: **https://claude.ai/artifact/5NMmksgZdVjbt2pUBf5tZK**
(as falas também estão nas notas de cada slide, no modo apresentador).

**8 slides · 5 min 47 s no cronômetro.** O enunciado dá até 7 minutos. Se quiser
fechar em **5:07**, corte o slide 5 — está marcado abaixo.

Existe também o [roteiro técnico](roteiro-video.md), com 12 slides e a linguagem
de arguição. Este aqui é a mesma coisa dita em voz alta: **a fala é simples e o
termo exato fica no pé do slide**, onde o avaliador lê sem que você precise
pronunciar. Os dois cobrem os mesmos nove tópicos exigidos.

| tópico exigido | slide |
|---|---|
| problema | 2 |
| arquitetura | 3 |
| estratégia de RAG | 3, 4 |
| recuperação | 3 |
| demonstração | 7 |
| exemplos | 5, 7 |
| evidências | 6, 7 |
| desafios | 4, 8 |
| decisões | 3, 4, 7, 8 |

---

## Antes de gravar

1. **Teste o áudio.** Mais vídeo é reprovado por áudio ruim que por conteúdo.

2. **Não grave a tela do terminal ao vivo.** A primeira resposta leva ~20 s
   porque carrega o modelo, e 20 s de silêncio custam 7% do vídeo. O slide 7 já
   tem a saída real impressa — use o slide e diga que é saída real. Se quiser
   conferir antes:

   ```bash
   python app.py -p "o que dizem os clientes sobre a caixa amassada?" --fontes
   ```

3. **Olhe para a câmera no slide 1 e no slide 8.** No meio, pode ler.

4. **Não leia o rodapé técnico em voz alta.** Ele existe justamente para você
   não ter que pronunciar "fusão por RRF com k igual a 60". Se a banca quiser o
   termo, ele está na tela.

---

## O roteiro, slide a slide

### 1 · Capa — 12 s

> Boa tarde. Eu construí um sistema que responde perguntas sobre as avaliações
> que os clientes de um e-commerce escreveram. São 99 mil avaliações.
>
> A frase que importa é a de baixo: **toda resposta vem junto com a avaliação
> que sustenta ela** — ou vem uma recusa, quando o sistema não tem base para
> responder. É isso que eu vou mostrar em cinco minutos.

---

### 2 · O problema — 40 s

> O problema é este. São 99 mil avaliações de clientes, de dois anos de loja. A
> nota de 1 a 5 diz **o quanto** o cliente ficou insatisfeito — mas só o texto
> diz **por quê**. E esse texto existe em 43% das avaliações, ou seja, quase
> metade. Depois da limpeza, 39 mil textos entraram na base do sistema.
>
> Ninguém lê 39 mil avaliações à mão. E banco de dados não resolve, porque a
> pergunta é de outro tipo: *"qual a nota média da categoria"* é uma consulta
> comum, qualquer sistema faz. *"O que o cliente reclama quando a entrega
> atrasa"* é uma pergunta sobre o que ele escreveu — e é aí que o meu sistema
> entra.

---

### 3 · Como funciona — 50 s

> A arquitetura tem três caminhos, e essa é a decisão de desenho mais
> importante do projeto.
>
> Se a pergunta é sobre **conteúdo** — *"o que reclamam da embalagem"* — ele
> busca as avaliações parecidas e responde citando elas. Se a pergunta é de
> **quantidade** — *"quais os principais problemas"* — ele conta a base inteira
> e devolve o ranking, e esse caminho não passa pelo modelo de linguagem. E se é
> só um **número** — *"qual a nota média de móveis"* — é cálculo direto na
> tabela, em um décimo de segundo.
>
> Quem decide o caminho é **uma regra que eu escrevi, não outro modelo**. Regra
> erra de forma previsível, e na dúvida ela manda para o caminho que sabe dizer
> "não sei".
>
> Sobre a busca em si: ela faz **duas buscas ao mesmo tempo**. Uma procura a
> palavra igual à da pergunta, a outra procura o sentido parecido. Depois o
> sistema fica com o que as duas concordam. Tudo isso roda no meu notebook, sem
> chave de API e sem nenhum dado sair da máquina.

---

### 4 · As duas armadilhas — 55 s

> Agora as duas armadilhas. São as duas decisões técnicas que definem o
> projeto, e as duas me custaram retrabalho.
>
> A primeira: o jeito padrão de limpar texto, o que se ensina em qualquer curso,
> joga fora as palavras muito comuns. E o **"não" é uma delas**. Olhem o
> exemplo: *"não chegou o produto"* e *"chegou o produto"* viram exatamente a
> mesma coisa depois da limpeza. Numa base cujo valor inteiro está nas
> reclamações, isso é destrutivo — e um quarto dos textos tem negação.
>
> E não é defeito do curso: é que um método **conta palavra**, onde o "não" é
> mesmo ruído, e o outro representa **sentido**, onde ele é tudo. Objetivos
> opostos. Por isso o projeto tem três rotas de limpeza em vez de uma.
>
> A segunda armadilha é mais sutil. Se eu pergunto *"quais os principais
> problemas"*, a busca me devolve as cinco avaliações mais **parecidas** com a
> palavra "problema". Parece uma resposta certa e está metodologicamente
> errada, porque **parecido não é frequente**. Por isso contar é um caminho
> separado, que olha a base inteira em vez de cinco exemplos.

---

### 5 · Os problemas, em ordem — 40 s · **CORTE AQUI SE PRECISAR**

> Esta é a resposta do caminho de contagem, e ela sai em um décimo de segundo. O
> problema mais relatado é **produto não recebido**, com 17% dos relatos.
>
> Repare numa coisa que eu fiz de propósito: o **denominador está escrito dentro
> da figura**. São 13 mil avaliações com problema identificado, não as 99 mil da
> base. Isso muda a frase: dizer *"32% dos clientes reclamam de entrega"* seria
> **falso**; dizer *"32% dos relatos de problema são de entrega"* é **verdade**.
> Gráfico de frequência sem denominador é exatamente o erro que esse caminho
> existe para evitar.

*Cortando este slide o vídeo fecha em 5:07. Diga só "17% são de produto não
recebido" na transição para o slide 6 — ele sustenta o argumento sozinho.*

---

### 6 · A prova — 45 s · **O SLIDE MAIS IMPORTANTE**

> Este é o slide que eu defenderia primeiro numa arguição, e vou explicar por
> quê.
>
> Os temas foram formados a partir do **texto** que o cliente escreveu. A coluna
> que vocês estão vendo — o percentual de pedidos que de fato atrasaram — vem
> das **datas do pedido**. São duas informações independentes: a data nunca
> participou da formação dos temas. E mesmo assim ela separa os grupos em dois
> blocos limpos.
>
> Olhem o detalhe que mais importa. *"Produto não recebido"* tem nota média 1,8.
> *"Quantidade errada"* tem nota 2,0. **Pela nota são indistinguíveis** —
> qualquer relatório diria que são o mesmo problema. Mas **pelo atraso real
> estão a dez vezes de distância**: 41% contra 4%.
>
> Ou seja: a camada de temas acrescenta um eixo que a nota não tem. E se a minha
> leitura do texto estivesse errada, essa coluna estaria embaralhada. Ela não
> está.

---

### 7 · Vendo funcionar — 55 s

> Esta é a saída **real**, não uma maquete.
>
> Eu pergunto sobre a caixa amassada. O sistema mostra o caminho que escolheu,
> responde em duas linhas, cita cinco avaliações pelo código e — isto é o mais
> importante — acrescenta uma **ressalva de evidência fraca**.
>
> E a ressalva **está certa**. Das cinco avaliações que ele recuperou, duas são
> nota 4 e 5: gente que recebeu a caixa amassada e o produto intacto. O sistema
> está dizendo *"existe base, mas é fraca"*, em vez de afirmar no mesmo tom de
> quando tem folga.
>
> Embaixo, cada código abre o depoimento inteiro, com data, nota, categoria e
> estado. É isso que torna o insight rastreável até a pessoa que escreveu.
>
> E ele tem **quatro travas** antes de afirmar qualquer coisa: a primeira
> bloqueia pergunta com CPF antes de buscar nada; a segunda mede se a evidência
> é boa o bastante, e se não é, **o modelo nem é chamado**; a terceira manda o
> modelo recusar; e a quarta confere, **por código**, se as avaliações que ele
> citou estavam de verdade no material — se inventou um código, a resposta é
> descartada. **Só a terceira depende de o modelo obedecer.**

---

### 8 · Os números e os limites — 55 s

> Os números, e eu fecho com os limites porque eles valem mais que a lista do
> que funciona.
>
> Na busca, **10 acertos em 12**, num gabarito que eu montei a partir dos temas.
> O portão de evidência foi ajustado em 22 perguntas e testado em **16 perguntas
> novas**, que nunca participaram da escolha do corte — e acertou 100% delas.
> **Invenção zero** na mediana das respostas. Custo zero por pergunta, porque
> tudo roda local, e 91 testes automatizados.
>
> Uma observação sobre o último número: eu **não usei a ferramenta padrão de
> avaliação** de RAG. Ela precisa de um modelo grande para julgar as respostas,
> e o meu modelo é pequeno — seria o modelo fraco se avaliando. Então medi outra
> coisa, mais simples e **conferível**: de onde vem cada palavra da resposta.
>
> E o limite principal, que sai escrito em **toda** resposta: só escreve
> avaliação quem ficou muito satisfeito ou muito irritado. Então o que eu conto
> é o que **essas pessoas escreveram**, nunca o que todos os clientes pensam.
>
> Termino com a ideia que eu levaria para qualquer projeto desses: **a correção
> deste sistema não depende da qualidade do modelo.** A conferência de citação
> compara texto, a contagem vem da tabela, o portão vem da busca. O modelo só
> redige a frase final. É por isso que ele roda num notebook sem placa de vídeo:
> um modelo pequeno escreve pior, **mas não cita falso**. Obrigado.

---

## O que eu digo ↔ o que está escrito no slide

Esta é a tradução que o deck usa. Se a banca perguntar pelo termo, ele já está
na tela — você só confirma.

| o que você fala | o termo no rodapé do slide |
|---|---|
| "duas buscas ao mesmo tempo: palavra igual e sentido parecido" | busca híbrida BM25 + FAISS |
| "ele fica com o que as duas concordam" | fusão por RRF, k=60 |
| "uma regra que eu escrevi, não outro modelo" | roteador por regra |
| "mede se a evidência é boa o bastante" | portão de suficiência, limiar 0,689 |
| "confere por código se ele citou de verdade" | verificação de `review_id` por comparação de string |
| "joga fora as palavras comuns" | remoção de stop words (NLTK pt) |
| "contar é um caminho separado" | camada de frequência: BERTopic, 101 tópicos → 16 temas |
| "de onde vem cada palavra da resposta" | métrica de ancoragem lexical (em vez de RAGAS) |
| "um modelo pequeno, no notebook" | Qwen2.5-0.5B-Instruct, local, sem chave de API |

---

## Se a banca perguntar depois

| pergunta provável | resposta curta |
|---|---|
| *"Por que não usou LangChain?"* | A abstração esconde justamente as peças que estão em avaliação: a fusão das buscas, o limiar, a montagem do contexto. E portão, termo obrigatório e conferência de citação não existem no framework — eu teria de escrever à mão de qualquer jeito. Está registrado em `decisoes.md`. |
| *"Por que um modelo tão pequeno?"* | Restrição de memória medida: 0,2 GB livres na máquina. E a troca não afeta a correção, porque a conferência de citação é comparação de texto. O modelo maior é selecionável por variável de ambiente. |
| *"Como sabe que não alucina?"* | Não sei que nunca alucina. Sei que **não cita falso**, porque isso é conferido por código. E medi invenção de palavra: mediana zero. Os limites da métrica estão declarados no README. |
| *"O filtro por categoria e estado funciona?"* | No caminho de contagem sim — categoria, estado, mês e atraso. No caminho de evidência **ainda não**, e está registrado como o próximo passo de maior retorno. |
| *"Por que BERTopic e não LDA?"* | Corpus de textos curtos em português. O LDA precisa de documento longo para estimar a mistura de tópicos, e aqui a mediana é 10 palavras. |
| *"E se a base crescer 100 vezes?"* | A busca atual é exata e viraria o gargalo. Trocaria pelo índice aproximado ou por `pgvector` — está citado como caminho de produção. A arquitetura não muda. |
