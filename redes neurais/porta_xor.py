import pandas as pd
import numpy as np
import math 
import random

#dados de entrada
#porta xor
x1 = [0,0,1,1]
x2 = [0,1,0,1]
classe = [0,1,0,1]

#pesos gerados aleatoriamente depois diminuir o codigo
peso1 = random.random()
peso2 = random.random()
peso3 = random.random()
peso4 = random.random()
print(peso1,peso2)

def funcao_soma():
    soma = (x1 * peso1) + (x2 * peso2)
    return soma

def funcao_sigmoide_ativacao():
    sigmoide = 1 / (1 + math.exp(-x))
    return sigmoide

#executando a rede neural, ou pelo menos "tentando"
def rede_neural():
    (x1[0] * peso1) +  (x2[0] * peso2)
    (x1[0] * peso1) +  (x2[0] * peso2)


neuronio_1 = (x1[0] * peso1) +  (x2[0] * peso2)
neuronio_2 = (x1[0] * peso1) +  (x2[0] * peso2)
print(neuronio_1,neuronio_2)