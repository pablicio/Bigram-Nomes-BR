# Gerador de Nomes Brasileiros com MLP em PyTorch

Este projeto � uma adapta��o do exerc�cio apresentado por **Andrej Karpathy** na aula  
**Building Makemore - Part 2: MLP**, onde foi implementado um gerador de nomes  
usando uma rede neural multicamada (MLP) em PyTorch.

O c�digo original foi ajustado para trabalhar com **nomes brasileiros** a partir  
de dados do IBGE, tanto masculinos quanto femininos.

---

## O que foi feito

1. **Coleta de dados**  
   - Foram utilizados dois arquivos CSV com nomes brasileiros mais comuns (masculinos e femininos) disponibilizados pelo IBGE.
   - Os arquivos foram combinados em um �nico dataset e pr�-processados para conter apenas os nomes.

2. **Pr�-processamento**  
   - Convers�o dos nomes para tokens (�ndices num�ricos).
   - Cria��o de um vocabul�rio (`stoi` e `itos`) mapeando letras para �ndices e vice-versa.
   - Gera��o de exemplos de entrada/sa�da para treino com *contexto fixo* (`block_size`).

3. **Defini��o do modelo**  
   - **Camada de embeddings** (`C`) para representar cada token como um vetor d-dimensional.
   - **Primeira camada totalmente conectada** (`W1`, `b1`) com ativa��o `tanh`.
   - **Camada de sa�da** (`W2`, `b2`) para previs�o da pr�xima letra.

4. **Treinamento**  
   - Uso de *backpropagation* com `requires_grad=True` para todos os par�metros.
   - Ajuste iterativo dos pesos para minimizar a *loss* de previs�o da pr�xima letra.

5. **Gera��o de nomes**  
   - In�cio com um contexto vazio (`[0] * block_size`).
   - Previs�o da pr�xima letra de forma probabil�stica (`torch.multinomial`).
   - Continua��o at� encontrar o token especial de fim (`0`).
   - Convers�o dos �ndices para letras e impress�o dos nomes gerados.

---

## Estrutura do C�digo

- **Carregamento de dados**: leitura e unifica��o dos CSVs do IBGE.
- **Tokeniza��o**: cria��o de mapeamentos `stoi` e `itos`.
- **Defini��o de par�metros**: inicializa��o manual dos pesos e vieses da MLP.
- **Treinamento**: c�lculo de *loss*, retropropaga��o e atualiza��o de par�metros.
- **Amostragem**: gera��o de novos nomes letra por letra.

---

## Cr�ditos

Este projeto � baseado e adaptado da aula:

**Building Makemore - Part 2: MLP**  
por **Andrej Karpathy**  

Dispon�vel no YouTube:  
[https://www.youtube.com/watch?v=TCH_1BHY58I](https://www.youtube.com/watch?v=TCH_1BHY58I)  

Notebook original no Google Colab:  
[https://colab.research.google.com/drive/1YIfmkftLrz6MPTOO9Vwqrop2Q5llHIGK?usp=sharing#scrollTo=TQUMmgRrdRIA](https://colab.research.google.com/drive/1YIfmkftLrz6MPTOO9Vwqrop2Q5llHIGK?usp=sharing#scrollTo=TQUMmgRrdRIA)  

---

## Fonte dos Nomes Brasileiros

Os nomes utilizados foram obtidos do reposit�rio [MedidaSP/nomes-brasileiros-ibge](https://github.com/MedidaSP/nomes-brasileiros-ibge), que cont�m:

- **`README.md`**  
  Breve descri��o do projeto, mencionando que se trata de uma lista dos nomes mais populares no Brasil por g�nero, extra�da do IBGE (com se��o *TODO* ainda n�o preenchida).

- **`ibge-fem-10000.csv`**  
  Lista com os 10.000 nomes femininos mais frequentes no Brasil, incluindo as colunas:  
  `nome`, `regiao`, `freq`, `rank`, `sexo`.

- **`ibge-mas-10000.csv`**  
  Lista com os 10.000 nomes masculinos mais frequentes no Brasil, com as mesmas colunas:  
  `nome`, `regiao`, `freq`, `rank`, `sexo`.

Esses arquivos fornecem uma listagem abrangente dos nomes mais comuns no pa�s, separados por g�nero, conforme dados do IBGE.

---

## Tecnologias Utilizadas

- Python 3
- PyTorch

---

## Observa��o

O modelo aqui apresentado � uma implementa��o **did�tica** para entender o  
funcionamento de embeddings, MLP e gera��o de sequ�ncias de texto.  
N�o � otimizado para produ��o, mas serve como base para experimentos.
