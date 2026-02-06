# Algoritmo Genético para Otimização da Função de Rosenbrock (Python)

Este repositório contém uma implementação de Algoritmo Genético em Python para minimizar a função de Rosenbrock (também conhecida como "banana function").

A solução inclui:
- Menu interativo para ajuste de parâmetros
- Seleção por torneio
- Cruzamento aritmético
- Mutação gaussiana com limitação de domínio
- Elitismo
- Geração automática de logs e gráficos por geração



## Função objetivo

A função utilizada é a Rosenbrock em duas variáveis:

f(x, y) = (1 - x)^2 + 100 (y - x^2)^2

Domínio (configurado no código):
- x ∈ [-2, 4]
- y ∈ [-2, 4]

Observação: o AG está configurado para minimizar a função (menor valor é melhor).


## Estratégia do Algoritmo Genético

Representação:
- Indivíduos representados diretamente como valores reais (x, y) em um array NumPy

Operadores:
- Seleção: torneio (tamanho configurável)
- Cruzamento: aritmético (mistura com fator beta aleatório)
- Mutação: gaussiana (normal com desvio padrão configurável), com recorte (clip) para manter o indivíduo dentro do domínio
- Elitismo: percentual configurável, garantindo pelo menos 1 indivíduo preservado


## Estrutura de saída

Ao executar, o programa cria a pasta:

resultados/
  log.txt
  fitness_evolucao_medio.png
  fitness_evolucao_melhor.png
  graficos/
    geracao_1.png
    geracao_2.png
    ...

- log.txt: registra estatísticas por geração (fitness médio, melhor fitness e melhor indivíduo)
- graficos/geracao_N.png: dispersão da população por geração, destacando o melhor indivíduo
- fitness_evolucao_medio.png: evolução do fitness médio por geração
- fitness_evolucao_melhor.png: evolução do melhor fitness por geração



## Como executar

Requisitos:
- Python 3.x
- numpy
- matplotlib

Instalação de dependências:

- pip install numpy matplotlib


Execução:

- python AG.py


O programa abre um menu que permite configurar:

- Tamanho da população

- Número de gerações

- Taxa de cruzamento

- Taxa de mutação

- Tamanho do torneio

- Percentual de elitismo

- Desvio padrão da mutação (sigma)

- Rodar o algoritmo

Parâmetros padrão
Os valores padrão definidos no código são:

- População: 200

- Gerações: 100

- Taxa de cruzamento: 0.8

- Taxa de mutação: 0.05

- Torneio: 3

- Elitismo: 0.005 (garante ao menos 1 indivíduo)

- Sigma da mutação: 0.1

Observações técnicas
- A população é inicializada por distribuição uniforme dentro dos limites do domínio.

- O cruzamento aritmético é aplicado por par, sujeito à taxa de cruzamento.

- A mutação é aplicada gene a gene; quando ocorre, um novo valor é amostrado de uma normal centrada no valor atual.

- O recorte (np.clip) garante que nenhum indivíduo ultrapasse os limites.

- Os gráficos são salvos automaticamente por geração, o que pode gerar muitos arquivos se o número de gerações for alto.
