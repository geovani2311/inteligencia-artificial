import math 
import random
import pandas as pd

# Porta XOR
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
classe = [0, 1, 1, 0]

# Função soma 
def funcao_soma(x1, peso1, x2, peso2):
    return (x1 * peso1) + (x2 * peso2)

# Função sigmoide ativação
def funcao_sigmoide_ativacao(x):
    return 1 / (1 + math.exp(-x))

# Lista para armazenar os dados
resultados = []

# Simulação
for i in range(4):
    peso1 = random.random()
    peso2 = random.random()
    
    soma = funcao_soma(x1[i], peso1, x2[i], peso2)
    ativacao = funcao_sigmoide_ativacao(soma)
    erro = classe[i] - ativacao
    
    resultados.append({
        "x1": x1[i],
        "x2": x2[i],
        #"peso1": round(peso1, 4),
        #"peso2": round(peso2, 4),
        "classe": classe[i],
        "saida": round(ativacao, 4),
        "erro": round(erro, 4)
    })

# Criar DataFrame
df_resultados = pd.DataFrame(resultados)

# Mostrar
print(df_resultados)