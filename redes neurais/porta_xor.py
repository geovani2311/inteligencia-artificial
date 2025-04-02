import pandas as pd
import numpy as np
import math 
import random

#dados de entrada
#x1 = [0,0,1,1]
#x1 = [0,1,0,1]
#classe = [0,1,0,1]

#pesos gerados aleatoriamente
peso1 = random.random()
peso2 = random.random()
print(peso1,peso2)

def funcao_soma():
    soma = (x1 * peso1) + (x2 * peso2)
    return soma

def funcao_sigmoide():
    sigmoide = 1 / (1 + math.exp(-x))
    return sigmoide

#def funcao_ativacao():
#    calculo = (x1 * peso1) + (x2 * peso2)
#    return calculo
