import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# configuracao e parametro global 
# ==============================================================================

# --- parametro da otimizacao  ---
LIMITES = [(-2, 4), (-2, 4)]  # (min_x, max_x), (min_y, max_y) funcao 12 
QTD_VARIAVEIS = 2 # usei pra deixar o codigo bonito ( n ficar usando 2 nas passagens)

# --- saida dos resultados, graficos e log ---
DIR_RESULTADOS = "resultados"
DIR_GRAFICOS = os.path.join(DIR_RESULTADOS, "graficos")
LOG_FILE = os.path.join(DIR_RESULTADOS, "log.txt")


# ==============================================================================
#    funcoes do algoritmo genetico
# ==============================================================================

def funcao_aptidao(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2 # funcao 12 - z= (1-x)^2 + 100(y-x^2)^2

def calcular_fitness_pop(populacao):
    x = populacao[:, 0] #pega todas as linhas da coluna 0 ou seja do gene x
    y = populacao[:, 1] #faz igual so q pra coluna 1 ou seja gene y
    return funcao_aptidao(x, y) # calcula todos os individuos da populacao ja q pegou todas as linhas 

def inicializar_populacao(tam_pop):

   #lembrar np.random.uniform(low, high, size)

    populacao = np.zeros((tam_pop, QTD_VARIAVEIS)) #ele cria uma matriz população, onde a quantidade de linhas é o tamanho da população e a qtd variaveis é a qntdade de genes
    populacao[:, 0] = np.random.uniform(LIMITES[0][0], LIMITES[0][1], tam_pop)  #pra deixar claro os indices dos limites, basicamente o primeiro indice representa qual gene, e o segundo indice se eh o maximo ou o minimo
    populacao[:, 1] = np.random.uniform(LIMITES[1][0], LIMITES[1][1], tam_pop)
    return populacao

def selecao_torneio(populacao, fitness, tam_pop, tam_torneio):

    #Seleciona N competidores aleatoriamente

    # np.random.choice = (do intervalo 0 até VALOR, pega VALOR individuos, sem ou com repetição)

    indices_competidores = np.random.choice(tam_pop, tam_torneio, replace=False)
    
    #  encontra o fitness desses competidores
    fitness_competidores = fitness[indices_competidores]
    


    #  os nomes deixaram confusos, mas basicamente o vencedor local e o vencedor global são o mesmo individuo
    #  o vencedor local, recebe o indice do individuo com maior fitness, porém esse indice é o dos competidores, tipo imagine q selecionou 3 competidores, o indice no local vai ser de 0,1 ou 2
    # no vencedor global, ele basicamente pega o indice em relação a população de verdade, exemplo pop tam 150, vai pegar indice de 0-149

    indice_vencedor_local = np.argmax(fitness_competidores)  
    indice_vencedor_global = indices_competidores[indice_vencedor_local]
    
    return populacao[indice_vencedor_global]

def cruzamento_aritmetico(pai1, pai2, taxa_cruzamento):
    
    #verifica se o cruzamento deve ocorrer
    if np.random.rand() > taxa_cruzamento:
        return pai1.copy(), pai2.copy() 
        
    beta = np.random.rand()  # Fator de mistura aleatorio U[0,1]
    
    filho1 = beta * pai1 + (1 - beta) * pai2 #obs: faz isso aq nos 2 genes 
    filho2 = beta * pai2 + (1 - beta) * pai1
    
    return filho1, filho2

def mutacao_gaussiana(individuo, taxa_mutacao, mutacao_sigma):
    
    for i in range(QTD_VARIAVEIS):
        # Verifica se a mutação deve ocorrer neste gene
        if np.random.rand() < taxa_mutacao:
            valor_atual = individuo[i]
            
            # Gera novo valor da distribuição Normal

            # np.random.normal : vai pegar um novo valor aleatorio de acordo com a distribuicao gaussiana, ele vai ver um valor perto da loc, e a intensidade de busca vai ser de acordo com a escala

            novo_valor = np.random.normal(loc=valor_atual, scale=mutacao_sigma)
            
            #  (clip) o valor para manter dentro das restrições
            min_lim, max_lim = LIMITES[i]
            novo_valor = np.clip(novo_valor, min_lim, max_lim) # se os valores estiverem fora dos limites, vai mandar pro limite
            
            individuo[i] = novo_valor
            
    return individuo

# ==============================================================================
# funcoes de saida
# ==============================================================================

def setup_diretorios():
    
    os.makedirs(DIR_GRAFICOS, exist_ok=True) #cria o diretorio se n existir
    # Limpa o arquivo de log anterior, se existir
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    print(f"Resultados serão salvos em: {os.path.abspath(DIR_RESULTADOS)}")

def registrar_log(geracao, fitness_medio, melhor_fitness, melhor_ind):
    #escreve linha a linha no arq de log

    log_linha = (
        f"Geração {geracao}: "
        f"Fitnesse Médio= {fitness_medio:.4f}, "
        f"Melhor Fitness= {melhor_fitness:.4f}, "
        f"Melhor x: {melhor_ind[0]:.4f}, "
        f"Melhor y: {melhor_ind[1]:.4f}.\n"
    )
    
    #"a" = append ou seja na ultima linha vai somando os logs linhas 
    with open(LOG_FILE, "a") as f:
        f.write(log_linha)

def plotar_populacao(geracao, populacao, melhor_ind, melhor_fitness):
    #Gera e salva o gráfico de dispersão da população.
    plt.figure(figsize=(10, 8))
    

    # membros comuns da população
    plt.scatter(
        populacao[:, 0], populacao[:, 1], 
        alpha=0.6, label="População"
    )
    
    #melhor membro
    plt.scatter(
        melhor_ind[0], melhor_ind[1], 
        color='red', s=100, 
        edgecolor='black', label=f"Melhor (Fit: {melhor_fitness:.2f})"
    )
    
    plt.title(f"Geração {geracao} | Espaço de Busca")
    plt.xlabel("Variável x")
    plt.ylabel("Variável y")
    plt.xlim(LIMITES[0][0], LIMITES[0][1])
    plt.ylim(LIMITES[1][0], LIMITES[1][1])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    caminho_arquivo = os.path.join(DIR_GRAFICOS, f"geracao_{geracao}.png")
    plt.savefig(caminho_arquivo)
    plt.close()

def plotar_evolucao(historico_fitness_medio):
    #Gera e salva o gráfico de linha da evolução do fitness médio
    plt.figure(figsize=(10, 6))
    plt.plot(historico_fitness_medio)
    plt.title("Evolução do Fitness Médio por Geração")
    plt.xlabel("Geração")
    plt.ylabel("Fitness Médio")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    caminho_arquivo = os.path.join(DIR_RESULTADOS, "fitness_evolucao_medio.png")
    plt.savefig(caminho_arquivo)
    plt.close()

def plotar_evolucao_melhor(historico_melhor_fitness):
    #Gera e salva o gráfico de linha da evolução do MELHOR fitness
    plt.figure(figsize=(10, 6))
    plt.plot(historico_melhor_fitness, color='blue')
    plt.title("Evolução do Melhor Fitness por Geração")
    plt.xlabel("Geração")
    plt.ylabel("Melhor Fitness")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    caminho_arquivo = os.path.join(DIR_RESULTADOS, "fitness_evolucao_melhor.png")
    plt.savefig(caminho_arquivo)
    plt.close()



# ==============================================================================
# funcao de execucao
# ==============================================================================

def rodar_ag(parametros):
     

    
    # Desempacota todos os parâmetros
    TAM_POPULACAO = parametros["TAM_POPULACAO"]
    NUM_GERACOES = parametros["NUM_GERACOES"]
    TAXA_CRUZAMENTO = parametros["TAXA_CRUZAMENTO"]
    TAXA_MUTACAO = parametros["TAXA_MUTACAO"]
    TAM_TORNEIO = parametros["TAM_TORNEIO"]
    ELITISMO_PCT = parametros["ELITISMO_PCT"]
    MUTACAO_SIGMA = parametros["MUTACAO_SIGMA"]
    
    #calcula QTD_ELITE com base nos parâmetros atuais
    QTD_ELITE = max(1, int(TAM_POPULACAO * ELITISMO_PCT))

    # preparar ambiente
    setup_diretorios()
    historico_fitness_medio = []
    historico_melhor_fitness = [] 
    
    #  geração 0 
    populacao = inicializar_populacao(TAM_POPULACAO)
    
    print("\nIniciando otimização...")
    print(f"Parâmetros de Execução:")
    print(f"  População: {TAM_POPULACAO}, Gerações: {NUM_GERACOES}, Elite: {QTD_ELITE} ({ELITISMO_PCT*100}%)")
    print(f"  Taxa Cruz.: {TAXA_CRUZAMENTO}, Taxa Mut.: {TAXA_MUTACAO}, Torneio: {TAM_TORNEIO}")
    print(f"  Desvio Padrão (Mutação): {MUTACAO_SIGMA}")
    print("---")
    
    # loop das Gerações
    for geracao in range(1, NUM_GERACOES + 1):
        
        #  calcular Aptidão
        fitness = calcular_fitness_pop(populacao)
        
        # coletar Estatísticas
        melhor_fitness = np.max(fitness)
        fitness_medio = np.mean(fitness)
        melhor_idx = np.argmax(fitness)
        melhor_individuo = populacao[melhor_idx] #retorna um vetor com todos os dados do individuo
        
        historico_fitness_medio.append(fitness_medio)
        historico_melhor_fitness.append(melhor_fitness) 
        
        #  saida
        registrar_log(geracao, fitness_medio, melhor_fitness, melhor_individuo)
        plotar_populacao(geracao, populacao, melhor_individuo, melhor_fitness)
        
        # imprime progresso no console pra feedback e tambem deixa mais bonito
        print(f"Geração {geracao:03d}/{NUM_GERACOES} | "
              f"Melhor Fitness: {melhor_fitness:10.2f} | "
              f"Médio: {fitness_medio:10.2f}")
        
        #  Preparar Próxima Geração
        nova_populacao = np.zeros_like(populacao)
        
        # elitismo
        indices_elite = np.argsort(fitness)[-QTD_ELITE:]
        nova_populacao[0:QTD_ELITE] = populacao[indices_elite]
        
        #loop de Reprodução
        for i in range(QTD_ELITE, TAM_POPULACAO, 2):
            # Seleção
            pai1 = selecao_torneio(populacao, fitness, TAM_POPULACAO, TAM_TORNEIO)
            pai2 = selecao_torneio(populacao, fitness, TAM_POPULACAO, TAM_TORNEIO)
            
            # Cruzamento
            filho1, filho2 = cruzamento_aritmetico(pai1, pai2, TAXA_CRUZAMENTO)
            
            # Mutação
            filho1 = mutacao_gaussiana(filho1, TAXA_MUTACAO, MUTACAO_SIGMA)
            filho2 = mutacao_gaussiana(filho2, TAXA_MUTACAO, MUTACAO_SIGMA)
            
            # Adiciona à nova população
            nova_populacao[i] = filho1
            
            #confere se não ta fora do limite pra n bugar qndo é população impar
            #  Se TAM_POPULACAO for ímpar, na última iteração i+1 será igual a TAM_POPULACAO   Python dava IndexError.
            if i + 1 < TAM_POPULACAO:
                nova_populacao[i+1] = filho2
                
        #  Atualizar População
        populacao = nova_populacao

    #  fim da Execução
    print("---")
    print("Otimização concluída.")
    
    #  gerar grafico Final
    plotar_evolucao(historico_fitness_medio)
    print(f"Gráfico de evolução (Fitness Médio) salvo em: {os.path.join(DIR_RESULTADOS, 'fitness_evolucao_medio.png')}")
    
    
    plotar_evolucao_melhor(historico_melhor_fitness)
    print(f"Gráfico de evolução (Melhor Fitness) salvo em: {os.path.join(DIR_RESULTADOS, 'fitness_evolucao_melhor.png')}")
    
    print("\nPressione Enter para voltar ao menu...")
    input()


# ==============================================================================
# menu e loop principal
# ==============================================================================

def mostrar_menu(parametros):
    
    os.system('cls' if os.name == 'nt' else 'clear') # Limpa a tela
    print("===============================================================")
    print("           ALGORITMO GENÉTICO - CONFIGURAÇÃO")
    print("===============================================================")
    print(f" 1. Tamanho da População : {parametros['TAM_POPULACAO']}")
    print(f" 2. Número de Gerações   : {parametros['NUM_GERACOES']}")
    print(f" 3. Taxa de Cruzamento   : {parametros['TAXA_CRUZAMENTO']:.2f}")
    print(f" 4. Taxa de Mutação      : {parametros['TAXA_MUTACAO']:.2f}")
    print(f" 5. Tamanho do Torneio   : {parametros['TAM_TORNEIO']}")
    
    # Calcula a contagem de elite para exibição
    elite_count = max(1, int(parametros['TAM_POPULACAO'] * parametros['ELITISMO_PCT']))
    print(f" 6. % Elitismo           : {parametros['ELITISMO_PCT']:.3f} ({elite_count} indivíduo(s))")
    
    print(f" 7. Desvio Padrão (Mut.) : {parametros['MUTACAO_SIGMA']:.2f}")
    print("---------------------------------------------------------------")
    print(" 8. Rodar Algoritmo")
    print(" 9. Sair")
    print("===============================================================")
    return input("Escolha uma opção (1-9): ")

def ler_int_validado(prompt, min_val=1):
    #Lê um inteiro do usuário e o valida.
    while True:
        try:
            valor = int(input(prompt))
            if valor >= min_val:
                return valor
            else:
                print(f"Erro: O valor deve ser no mínimo {min_val}.")
        except ValueError:
            print("Erro: Por favor, digite um número inteiro válido.")

def ler_float_validado(prompt, min_val=0.0, max_val=None):
   
   # le um float do usuário e o valida.
   #permite max_val=None para parâmetros sem limite superior (como sigma).
    
    while True:
        try:
            valor = float(input(prompt))
            if valor < min_val:
                print(f"Erro: O valor deve ser no mínimo {min_val}.")
            elif max_val is not None and valor > max_val:
                print(f"Erro: O valor deve ser no máximo {max_val}.")
            else:
                return valor
        except ValueError:
            print("Erro: Por favor, digite um número flutuante válido (ex: 0.5).")

def main():
    #função principal 
    
    # Parâmetros padrão default
    parametros_atuais = {
        "TAM_POPULACAO": 200,
        "NUM_GERACOES": 100,
        "TAXA_CRUZAMENTO": 0.8,
        "TAXA_MUTACAO": 0.05,
        "TAM_TORNEIO": 3,
        "ELITISMO_PCT": 0.005,
        "MUTACAO_SIGMA": 0.1
    }
    
    while True:
        escolha = mostrar_menu(parametros_atuais)
        
        if escolha == '1':
            parametros_atuais["TAM_POPULACAO"] = ler_int_validado(
                "Novo Tamanho da População (ex: 200, mín. 10): ", min_val=10
            )
        
        elif escolha == '2':
            parametros_atuais["NUM_GERACOES"] = ler_int_validado(
                "Novo Número de Gerações (ex: 100, mín. 1): ", min_val=1
            )
            
        elif escolha == '3':
            parametros_atuais["TAXA_CRUZAMENTO"] = ler_float_validado(
                "Nova Taxa de Cruzamento (0.0 a 1.0): ", max_val=1.0
            )
            
        elif escolha == '4':
            parametros_atuais["TAXA_MUTACAO"] = ler_float_validado(
                "Nova Taxa de Mutação (0.0 a 1.0): ", max_val=1.0
            )
        
        elif escolha == '5':
            parametros_atuais["TAM_TORNEIO"] = ler_int_validado(
                "Novo Tamanho do Torneio (ex: 3, mín. 2): ", min_val=2
            )

        elif escolha == '6':
            parametros_atuais["ELITISMO_PCT"] = ler_float_validado(
                "Novo Percentual de Elitismo (0.0 a 1.0, ex: 0.01): ", max_val=1.0
            )
            
        elif escolha == '7':
            parametros_atuais["MUTACAO_SIGMA"] = ler_float_validado(
                "Novo Desvio Padrão da Mutação (ex: 0.1, mín. 0.01): ", min_val=0.01
            )
            
        elif escolha == '8':
            # Roda o AG com os parâmetros atuais
            rodar_ag(parametros_atuais)
            
        elif escolha == '9':
            print("Saindo...")
            break
            
        else:
            print("Opção inválida. Pressione Enter para tentar novamente.")
            input()

if __name__ == "__main__":
    main()