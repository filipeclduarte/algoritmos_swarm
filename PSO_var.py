####### Algoritmo PSO com Decaimento de Inércia Li e Gao (2009) ########
####### Algoritmo PSO ########

# Importando as bibliotecas básicas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from funcoes_benchmark import ackley, alpine, schwefel, happy_cat, brown, exponential_function

# Criar a função do PSO
def PSO_var(fun, *args, qtd_particulas, atributos_dim, min, max, seed = np.random.seed(1), max_epoch, w_in, w_fim, c1, c2):
    '''
        Função do Algoritmo SWARM PSO. 
        Inputs:
        - fun_opt: Função de fitness a ser otimizada
        - qtd_particulas: Quantidade de partículas
        - atributos_dim: Dimensão do Vetor de atributos 
        - min: intervalo inferior do domínio da função  
        - max: intervalo superior do domínio da função
        - seed: por padrão np.random.seed(1)
        - w_in: inércia inicial
        - w_fim: inércia final  
        - c1: influência do pbest (termo cognitivo)
        - c2: influência do gbest (termo do aprendizado social)
    '''
    
    ## Weight decay
    d1 = 0.2
    d2 = 7
    
    def weight_decay(w_in, w_fim, d1, d2, iter, iter_max):
        return (w_in - w_fim - d1) * np.exp(1/(1 + (d2 * iter/iter_max)))

    # inicializar as partículas em posições aleatórias
    particulas = np.random.uniform(low = min, high = max, size = (qtd_particulas, atributos_dim))
    #print('Partículas: \n', particulas)

    # inicializar a velocidade
    #velocidade = np.random.uniform(low = min, high = max, size = (qtd_particulas, atributos_dim))
    velocidade = np.zeros((qtd_particulas, atributos_dim))

    # inicializar o pbest em zero
    #pbest = np.ones((qtd_particulas, atributos_dim))
    pbest = np.zeros((qtd_particulas,atributos_dim))

    gbest_value = np.inf
    gbest = 0
    
    # Extrair a posição do gbest 
    for z in np.arange(qtd_particulas):
        new_gbest_value = fun(particulas[z,:])
        if new_gbest_value < gbest_value:
            gbest = z
    
    gbest_value = particulas[gbest,:]
    #print('Valor da função no gbest:\n', gbest_value)

    funcao_iteracao = np.zeros(max_epoch)
    media = np.zeros(max_epoch)
    desvio_pad = np.zeros(max_epoch)

    for k in np.arange(max_epoch):   
        w = weight_decay(w_in, w_fim, d1, d2, k, max_epoch)
        #print('epoch n.:', k)
        #print('\n')
    # Iterar para atualizar o pbest e gbest para cada partrícula
        for j in np.arange(qtd_particulas):
            if fun(particulas[j,:]) < fun(pbest[j,:]):
                pbest[j,:] = particulas[j,:]

                if fun(particulas[j,:]) < fun(particulas[gbest, :]):
                    gbest = j
                    gbest_value = fun(particulas[gbest, :])
                    #print('--------------------------------------')
                    #print('gbest:', gbest)
                    #print('Valor da função no gbest:', gbest_value)
                    #print('--------------------------------------')

            # Iteração para atualizar as posições das partículas
            for i in np.arange(qtd_particulas):
                r1, r2 = np.random.rand(), np.random.rand()
                velocidade[i, :] = w * velocidade[i, :] + c1 * r1 * (pbest[i, :] - particulas[i, :]) + c2 * r2 * (particulas[gbest, :] - particulas[i, :])
                particulas[i, :] = particulas[i, :] + velocidade[i, :]

        funcao_iteracao[k] = fun(particulas[gbest, :])
        
        vetor_fitness = np.zeros(qtd_particulas)

        for par in np.arange(qtd_particulas):
            vetor_fitness[par] = fun(particulas[par,:])
        
        media[k] = vetor_fitness.mean()
        desvio_pad[k] = vetor_fitness.std()

    return particulas, gbest, funcao_iteracao, media, desvio_pad


##### TESTE DO ALGORITMO PSO com decaimento da inércia ######
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
        part, melhor_part, funcao_iteracao, media, desvio_pad = PSO_var(fun = funcoes['funcoes'][func], qtd_particulas = 20,  atributos_dim = i, min = funcoes['dom_min'][func], max = funcoes['dom_max'][func], max_epoch = epocas, w_in = 0.95, w_fim = 0.4, c1 = 0.9, c2 = 0.8)

        # Mostrar resultados
        print('------------------------------------------------')
        print('Melhores Partículas :\n', part[melhor_part, :])
        print('\n Valor da função com as melhores partículas:', funcoes['funcoes'][func](part[melhor_part,:]))
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
        nome = "PSO_var_resultados/PSO_var_fig_"+str(str(funcoes['funcoes'][func])[10:16])+"_"+str(i)+'.png'
        plt.savefig(nome)
        plt.close()

        np.savetxt("PSO_var_resultados/PSO_var_"+str(str(funcoes['funcoes'][func])[10:16])+"_"+str(i)+".csv", part[melhor_part, :], delimiter=",")
        resultados.to_csv("PSO_var_resultados/PSO_var_resultados_"+str(str(funcoes['funcoes'][func])[10:16])+"_"+str(i)+".csv")