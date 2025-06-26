import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

df_portland = pd.read_csv('data/Portland_housePrices.csv', index_col=None)
print(df_portland.columns.tolist())


########################################################
# Função de Divisão dos Dados em Treino, Teste e Validação
########################################################
def data_splitting(df):
    train_test_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_test_df, validation_df

########################################################
# Modelo de Regressão Linear com Cross-Validation (K-Fold)
########################################################
def model(df, x_column, y_column):
    X = df[[x_column]]
    y = df[[y_column]]

    reg = LinearRegression()

    cv_scores = cross_val_score(reg, X, y, cv=5, scoring='r2')

    reg = LinearRegression().fit(X,y)

    return reg, cv_scores

########################################################
# Criação do Modelo Baseado no Tamanho
########################################################
def size_based_model_train(df):
    print(df)
    reg, cv_scores = model(df, 'tamanho', 'preco')
    return reg, cv_scores

########################################################
# Criação do Modelo Baseado no Número de Quartos
########################################################
def rooms_based_model_train(df):
    reg, cv_scores = model(df, 'quartos', 'preco')
    return reg, cv_scores

########################################################
# Plotagem do Modelo
########################################################
def plot_model(X, y, reg):

    x_label = X.columns[0]
    y_label = y.columns[0] 

    plt.scatter(X, y,color='g')
    plt.plot(X, reg.predict(X),color='k')
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def calc_rss(y,predicted):
    return ((predicted - y) ** 2).sum()

def calc_r2(y,predicted):
    return r2_score(predicted,y)

########################################################
# Função Principal
########################################################
if __name__ == "__main__":

    train_test_df, validation_df = data_splitting(df_portland)
    reg_size, cv_scores_size = size_based_model_train(train_test_df)
    reg_rooms, cv_scores_rooms = rooms_based_model_train(train_test_df)

    ########################################################
    # CV Score permite avaliar a qualidade do modelo, em cada fold de test
    # Quanto maior o score, melhor o modelo
    # O score é uma medida de quanto o modelo consegue prever
    # os dados de teste
    ########################################################
    print(f"CV scores for size-based model: {cv_scores_size}")
    print(f"CV scores for rooms-based model: {cv_scores_rooms}")

    plot_model(train_test_df[['tamanho']], train_test_df[['preco']], reg_size)
    plt.show()

    plot_model(train_test_df[['quartos']], train_test_df[['preco']], reg_rooms)
    plt.show()

    rss_size = calc_rss(validation_df[['preco']], reg_size.predict(validation_df[['tamanho']]))
    rss_rooms = calc_rss(validation_df[['preco']], reg_rooms.predict(validation_df[['quartos']]))

    r2_size = calc_r2(validation_df[['preco']], reg_size.predict(validation_df[['tamanho']]))
    r2_rooms = calc_r2(validation_df[['preco']], reg_rooms.predict(validation_df[['quartos']]))

    print(f"RSS for size-based model: {rss_size}")
    print(f"RSS for rooms-based model: {rss_rooms}")

    print(f"R2 for size-based model: {r2_size}")
    print(f"R2 for rooms-based model: {r2_rooms}")



