####### Algoritmo ABC guiado pelo GBEST Zhu e Kwong (2010) ########

####### Algoritmo ABC (Artificial Bee Colony) ########

# Importando as bibliotecas básicas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from funcoes_benchmark import ackley, alpine, schwefel, happy_cat, brown, exponential_function

# função de fitness
def fitness(fun, x):
    if fun(x) >= 0:
        return 1/(1+fun(x))
    else:
        return 1 + np.abs(fun(x))


# Criar a função do ABC guiado pelo gbest
def ABC_var(fun, fitness, *args, tam_colonia = 20, atributos_dim, min, max, seed = 1, max_cycle, C = 1.5):
    '''
     Função do Algoritmo SWARM ABC 
        Inputs:
        - qtd_abelhas: quantidade de abelhas, por padrão = 20
        - atributos_dim: dimensão do vetor 
        - min: valor mínimo do domínio da função (limite inferior)
        - max: valor máximo do domínio da função (limite superior)
        - seed: por padrão é np.random.seed(1), semente para garantir reprodutibilidade
        - max_cycle: quantidade de ciclos (iterações)
        - C: parâmetro limite da influência do gbest, por padrão = 1.5. 

    Três tipos de abelhas: Tamanho da colônia 
        empregadas (employed)
        exploradoras (scout) - apenas uma por ciclo
        seguidoras (onlookers)
        50% empregadas e 50% exploradoras 

    Fontes de alimentos: qtd_abelhas / 2
    
    Limite até abandonar uma fonte = (tam_colonia / 2) * atributos_dim
    '''
    # Seed
    np.random.seed(seed)

    # Parâmetros
    NP = int(tam_colonia) # tamanho da colônia
    D = int(atributos_dim) # tamanho da dimensão do problema
    limite_fonte = int((NP/2) * D) # limite da fonte de alimentos
    quantidade_alimentos = int(NP / 2) # quantidade de alimentos
    
    ## Inicialização
    ### Determinação aleatória das fontes
    fonte_alimentos = min + np.random.uniform(low = 0.0, high = 1.0, size = (NP, D)) * (max - min)

    ### Criação de um vetor para receber os valores dos fitness (nectar_amount)
    nectar_amount = np.zeros((NP, 1)) 

    ### Determinação do fitness das soluções aleatórias
    for fonte in np.arange(len(fonte_alimentos)):
        nectar_amount[fonte] = fitness(fun, fonte_alimentos[fonte, :])

    ### Posição da Melhor solução (k)
    melhor_solucao = np.argmax(nectar_amount)
    #print('Melhor solução gerada aleatoriamente na inicialização:\n', nectar_amount[melhor_solucao])

    # vetor de tentativas = 0
    limite = np.zeros((NP, 1))

    # Criando vetor de probabilidades = 0
    probs = np.zeros((NP, 1))

    # Ciclos até max_cycle
    funcao_iteracao = np.zeros(max_cycle)
    media = np.zeros(max_cycle)
    desvio_pad = np.zeros(max_cycle)

    for i in np.arange(max_cycle):
    #for i in np.arange(max_cycle):
    #    print("Ciclo:", i)
    #    print("---------------------------")
        
        # Fase das Abelhas Empregadas
        for j in np.arange(NP):
            
            # aplicação da eq. 2 de Karaboga - para cada abelha empregar a eq. v = xi + r * (xi - xk) com r e k randomicos
            r = np.random.uniform(low=-1.0, high=1.0) # aleatório entre -1 e 1
            fi = np.random.uniform(low = 0.0, high = C) # aleatório entre 0 e C
            k = np.random.choice([i for i in np.arange(quantidade_alimentos) if i != j], ) # selecionar uma fonte aleatoriamente (diferente de j)
            v = fonte_alimentos[j, :] + r * (fonte_alimentos[j, :] - fonte_alimentos[k, :]) + fi * (fonte_alimentos[melhor_solucao,:] - fonte_alimentos[j, :])

            # avaliar a função de fitness nessa fonte
            if (fitness(fun, fonte_alimentos[j, :]) < fitness(fun, v)):
            #if (fitness(fun, fonte_alimentos[j, :]) > nectar_amount[melhor_solucao]):
                fonte_alimentos[j, :] = v                
                nectar_amount[j] = fitness(fun, fonte_alimentos[j, :])
                limite[j] = 0
            else:
                limite[j] += 1

            # Computar a melhor solução se limite estourar, computar a melhor solução e voltar para o loop
            melhor_solucao = np.argmax(nectar_amount)
            #print('Melhor Solução:\n')
            #print(nectar_amount[melhor_solucao])
            
        # Fase das Abelhas Seguidoras (onlookers)
        ### Calcular as probabilidades
        #sum_fitness = 0
        sum_fitness = np.sum([fitness(fun, fonte_alimentos[f, :]) for f in np.arange(NP)])
        #for f in np.arange(NP):
        #    fit_f = fitness(fun, fonte_alimentos[f, :])
        #    print('Valor fit_f:', fit_f)
        #    sum_fitness += fit_f 
        #print('Sum Fitness for NP:', sum_fitness)
        
        for f in np.arange(NP):
            probs[f] = fitness(fun, fonte_alimentos[f, :]) / sum_fitness 
            # probs[f] = fitness(fun, fonte_alimentos[f, :]) / fitness(fun, fonte_alimentos).sum(axis = 0)) # probabilidades  = fit / sum(fit)
        #print('Vetor de probabilidades calculado:\n')
        #print(probs)
        for o in np.arange(NP):
            r1 = np.random.rand() # sortear um número aleatório
            #print('Número aleatório sorteado:', r1)
            # avaliar se o número é menor do que o pi - Se for, a fonte será escolhida
            if (r1 < probs[o]):
                r = np.random.uniform(low=-1.0, high=1.0) # aleatório entre -1 e 1
                fi = np.random.uniform(low = 0.0, high = C) # aleatório entre 0 e C
                k = np.random.choice([i for i in np.arange(quantidade_alimentos) if i != o], ) # selecionar uma fonte aleatoriamente (diferente de o)
                v = fonte_alimentos[o, :] + r * (fonte_alimentos[o, :] - fonte_alimentos[k, :]) + fi * (fonte_alimentos[melhor_solucao, :] - fonte_alimentos[o, :])

                # Avaliar o fitness da fonte:
                if(fitness(fun, fonte_alimentos[o, :]) < fitness(fun, v)):
                #if(fitness(fun, fonte_alimentos[o, :]) > nectar_amount[melhor_solucao]):
                    fonte_alimentos[o, :] = v                
                    nectar_amount[o] = fitness(fun, fonte_alimentos[o, :])
                    limite[o] = 0
                else:
                    limite[o] += 1
                
            # Computar a melhor solução se limite estourar, computar a melhor solução e voltar para o loop
            melhor_solucao = np.argmax(nectar_amount)

        # Fase das Abelhas Exploradoras (Scouts)
        for scout in np.arange(NP):
            if(limite[scout] > limite_fonte):
                fonte_alimentos[scout] = min + np.random.uniform(low = 0.0, high = 1.0, size = (1, D)) * (max - min)
                nectar_amount[scout] = fitness(fun, fonte_alimentos[scout, :])
                limite[scout] = 0 

        melhor_solucao = np.argmax(nectar_amount)
    
        funcao_iteracao[i] = fun(fonte_alimentos[melhor_solucao, :])
        
        vetor_fitness = np.zeros(NP)

        for font in np.arange(NP):
            vetor_fitness[font] = fun(fonte_alimentos[font,:])
        
        media[i] = vetor_fitness.mean()
        desvio_pad[i] = vetor_fitness.std()
        
    return fonte_alimentos, melhor_solucao, funcao_iteracao, media, desvio_pad


##### TESTE DO ALGORITMO ABC ######
dims = [10, 30, 50]
funcoes = [ackley, alpine, schwefel, happy_cat, brown, exponential_function]
dom_min = [-32.0, 0.0, -500.0, -2.0, -1.0, -1.0]
dom_max = [32.0, 10.0, 500.0, 2.0, 4.0, 1.0]

funcoes = {'funcoes':[ackley, alpine, schwefel, happy_cat, brown, exponential_function], 
           'dom_min': [-32.0, 0.0, -500.0, -2.0, -1.0, -1.0],
           'dom_max': [32.0, 10.0, 500.0, 2.0, 4.0, 1.0]}


epocas = 30

for i in dims:
    print('\nDIMENSÃO: {}\n'.format(i))
    for func in range(len(funcoes['funcoes'])):
        print('\nFUNÇÃO: {}\n'.format(funcoes['funcoes'][func]))
        
        # Rodando o algoritmo
        fonte_alimentos, melhor_solucao, funcao_iteracao, media, desvio_pad = ABC_var(fun = funcoes['funcoes'][func], fitness = fitness, tam_colonia = 100, atributos_dim = i, 
        min = funcoes['dom_min'][func], max = funcoes['dom_max'][func], seed = 1, max_cycle = epocas, C = 1.5)

        # Mostrar resultados
        print('------------------------------------------------')
        print('Melhor Fonte de alimentos :\n', fonte_alimentos[melhor_solucao, :])
        print('\n Valor da função com a melhor fonte de alimentos:', funcoes['funcoes'][func](fonte_alimentos[melhor_solucao,:]))
        print('\n------------------------------------------------')

         # Salvar resultados no dataframe
        resultados = {'fitness_iteracao': funcao_iteracao,
                      'media': media, 
                      'desvio_pad': desvio_pad}

        resultados = pd.DataFrame(resultados)


        # Plotar o gráfico da função em relação à iteração
        iteracao = np.array(np.arange(epocas))
        plt.plot(iteracao, funcao_iteracao)
        plt.title('Função no gbest ao longo das iterações - Função:'+str(str(funcoes['funcoes'][func])[10:16])+" dimensões: "+str(i))
        nome = "ABC_var_resultados/ABC_var_fig_"+str(str(funcoes['funcoes'][func])[10:16])+"_"+str(i)+'.png'
        plt.savefig(nome)
        plt.close()

        np.savetxt("ABC_var_resultados/ABC_var_"+str(str(funcoes['funcoes'][func])[10:16])+"_"+str(i)+".csv", fonte_alimentos[melhor_solucao, :], delimiter=",")
        resultados.to_csv("ABC_var_resultados/ABC_var_resultados_"+str(str(funcoes['funcoes'][func])[10:16])+"_"+str(i)+".csv")
