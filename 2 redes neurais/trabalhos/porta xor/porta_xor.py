import math
import random
import pandas as pd

# Porta XOR
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
classe = [0, 1, 1, 0]

# parâmetros
taxa_aprendizado = 0.9
epocas = 100

# pesos iniciais
peso1 = random.random()
peso2 = random.random()

# função soma
def funcao_soma(x1, peso1, x2, peso2):
    return (x1 * peso1) + (x2 * peso2)

# função sigmoide ativação
def funcao_sigmoide_ativacao(x):
    return 1 / (1 + math.exp(-x))

# derivada da função sigmoide
def derivada_sigmoide(x):
    return x * (1 - x)

# treinamento da rede neural
for epoca in range(epocas):
    for i in range(4):
        soma = funcao_soma(x1[i], peso1, x2[i], peso2)
        ativacao = funcao_sigmoide_ativacao(soma)
        erro = classe[i] - ativacao
        gradiente = derivada_sigmoide(ativacao)

        # atualização dos pesos
        peso1 += taxa_aprendizado * erro * gradiente * x1[i]
        peso2 += taxa_aprendizado * erro * gradiente * x2[i]

# aqui eu to salvando pra dentro de uma lista os resultados e depois transformando em um dataframe pra ficar mais bonitinho e fácil de ler e também arredondando os valores pra 4 casas decimais
resultados = []
for i in range(4):
    soma = funcao_soma(x1[i], peso1, x2[i], peso2)
    ativacao = funcao_sigmoide_ativacao(soma)
    erro = classe[i] - ativacao
    resultados.append({
        "x1": x1[i],
        "x2": x2[i],
        "classe": classe[i],
        "saida": round(ativacao, 4),
        "erro": round(erro, 4)
    })

#mostrando os resultados
df_resultados = pd.DataFrame(resultados)
print(df_resultados)

#tentar entender o que ta acontecendo pois está dando a mesma saida pra todas as vezes que roda o código e não ta dando a saida esperada, que seria 0, 1, 1, 0, depois de treinar a rede neural