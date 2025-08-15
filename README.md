# Gerador de Nomes Brasileiros com MLP em PyTorch

Este projeto é uma adaptação do exercício apresentado por **Andrej Karpathy** na aula  
**Building Makemore - Part 2: MLP**, onde foi implementado um gerador de nomes  
usando uma rede neural multicamada (MLP) em PyTorch.

O código original foi ajustado para trabalhar com **nomes brasileiros** a partir  
de dados do IBGE, tanto masculinos quanto femininos.

---

## O que foi feito

1. **Coleta de dados**  
   - Foram utilizados dois arquivos CSV com nomes brasileiros mais comuns (masculinos e femininos) disponibilizados pelo IBGE.
   - Os arquivos foram combinados em um único dataset e pré-processados para conter apenas os nomes.

2. **Pré-processamento**  
   - Conversão dos nomes para tokens (índices numéricos).
   - Criação de um vocabulário (`stoi` e `itos`) mapeando letras para índices e vice-versa.
   - Geração de exemplos de entrada/saída para treino com *contexto fixo* (`block_size`).

3. **Definição do modelo**  
   - **Camada de embeddings** (`C`) para representar cada token como um vetor d-dimensional.
   - **Primeira camada totalmente conectada** (`W1`, `b1`) com ativação `tanh`.
   - **Camada de saída** (`W2`, `b2`) para previsão da próxima letra.

4. **Treinamento**  
   - Uso de *backpropagation* com `requires_grad=True` para todos os parâmetros.
   - Ajuste iterativo dos pesos para minimizar a *loss* de previsão da próxima letra.

5. **Geração de nomes**  
   - Início com um contexto vazio (`[0] * block_size`).
   - Previsão da próxima letra de forma probabilística (`torch.multinomial`).
   - Continuação até encontrar o token especial de fim (`0`).
   - Conversão dos índices para letras e impressão dos nomes gerados.

---

## Estrutura do Código

- **Carregamento de dados**: leitura e unificação dos CSVs do IBGE.
- **Tokenização**: criação de mapeamentos `stoi` e `itos`.
- **Definição de parâmetros**: inicialização manual dos pesos e vieses da MLP.
- **Treinamento**: cálculo de *loss*, retropropagação e atualização de parâmetros.
- **Amostragem**: geração de novos nomes letra por letra.

---

## Créditos

Este projeto é baseado e adaptado da aula:

**Building Makemore - Part 2: MLP**  
por **Andrej Karpathy**  

Disponível no YouTube:  
[https://www.youtube.com/watch?v=TCH_1BHY58I](https://www.youtube.com/watch?v=TCH_1BHY58I)  

Notebook original no Google Colab:  
[https://colab.research.google.com/drive/1YIfmkftLrz6MPTOO9Vwqrop2Q5llHIGK?usp=sharing#scrollTo=TQUMmgRrdRIA](https://colab.research.google.com/drive/1YIfmkftLrz6MPTOO9Vwqrop2Q5llHIGK?usp=sharing#scrollTo=TQUMmgRrdRIA)  

---

## Fonte dos Nomes Brasileiros

Os nomes utilizados foram obtidos do repositório [MedidaSP/nomes-brasileiros-ibge](https://github.com/MedidaSP/nomes-brasileiros-ibge), que contém:

- **`README.md`**  
  Breve descrição do projeto, mencionando que se trata de uma lista dos nomes mais populares no Brasil por gênero, extraída do IBGE (com seção *TODO* ainda não preenchida).

- **`ibge-fem-10000.csv`**  
  Lista com os 10.000 nomes femininos mais frequentes no Brasil, incluindo as colunas:  
  `nome`, `regiao`, `freq`, `rank`, `sexo`.

- **`ibge-mas-10000.csv`**  
  Lista com os 10.000 nomes masculinos mais frequentes no Brasil, com as mesmas colunas:  
  `nome`, `regiao`, `freq`, `rank`, `sexo`.

Esses arquivos fornecem uma listagem abrangente dos nomes mais comuns no país, separados por gênero, conforme dados do IBGE.

---

## Tecnologias Utilizadas

- Python 3
- PyTorch

---

## Observação

O modelo aqui apresentado é uma implementação **didática** para entender o  
funcionamento de embeddings, MLP e geração de sequências de texto.  
Não é otimizado para produção, mas serve como base para experimentos.
