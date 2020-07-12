### Testar com 10, 20 e 50 dimensões (atributos)

### Funções benchmark 

import numpy as np


#### Funções: 
# Ackley function
def ackley(x):
    '''
    domínio da função: [-32.0, 32.0]
    '''
    x = x.reshape(-1, 1)
    n = x.shape[0]
    a,b,c = 20, 0.2, 2*np.pi
    return -a * np.exp(-b * np.sqrt((1/n) * (x**2).sum()) - np.exp((1/n) *(np.cos(c*x)).sum())) + a + np.exp(1)
    
# Alpine function
def alpine(x):
    '''
    domínio da função: [0., 10.]
    '''
    return np.abs((x * np.sin(x)) + (0.1 * x)).sum()

# Schwefel Function
def schwefel(x):
    '''
    domínio da função: [-500, 500]
    '''
    n = x.shape[0]
    return (418.9829 * n) - (x*np.sin(np.sqrt(np.abs(x)))).sum()

# Happy Cat Function
def happy_cat(x):
    '''
    domínio da função: [-2, 2]
    '''
    alfa = 1/8
    x = x.reshape(-1,1)
    n = x.shape[0]
    x2 = ((x * x).sum())**2
    return ((x2-n)**2)**alfa + (.5 * x2 + x.sum())/n + .5
    
# Brown function
def brown(x):
    '''
    domínio da função: [-1,4]
    '''
    x = x.reshape(-1,1)
    n = x.shape[0]
    scores = 0

    x = x**2
    for i in np.arange(n-1):
        scores = scores + x[i]**(x[i+1] + 1) + x[i+1] ** (x[i]+1)
    return scores

# Exponential function
def exponential_function(x):
    '''
    domínio da função: [-1,1]
    '''
    return -np.exp(-0.5 * (x**2).sum())