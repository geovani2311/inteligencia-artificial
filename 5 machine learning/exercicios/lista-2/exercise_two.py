import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

df_advertising = pd.read_csv('data/Advertising.csv', header=0, names=['tv', 'radio', 'newspaper', 'sales'])
print(df_advertising.head())

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
# Criação do Modelo Baseado em TV
########################################################
def tv_based_model_train(df):
    reg, cv_scores = model(df, 'tv', 'sales')
    return reg, cv_scores

########################################################
# Criação do Modelo Baseado em Radio
########################################################
def radio_based_model_train(df):
    reg, cv_scores = model(df, 'radio', 'sales')
    return reg, cv_scores

########################################################
# Criação do Modelo Baseado em Newspaper
########################################################
def newspaper_based_model_train(df):
    reg, cv_scores = model(df, 'newspaper', 'sales')
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

    train_test_df, validation_df = data_splitting(df_advertising)
    reg_tv, cv_scores_tv = tv_based_model_train(train_test_df)
    reg_radio, cv_scores_radio = radio_based_model_train(train_test_df)
    reg_newspaper, cv_scores_newspaper = newspaper_based_model_train(train_test_df)

    ########################################################
    # CV Score permite avaliar a qualidade do modelo, em cada fold de test
    # Quanto maior o score, melhor o modelo
    # O score é uma medida de quanto o modelo consegue prever
    # os dados de teste
    ########################################################
    print(f"CV scores for tv-based model: {cv_scores_tv}")
    print(f"CV scores for radio-based model: {cv_scores_radio}")
    print(f"CV scores for newspaper-based model: {cv_scores_newspaper}")

    plot_model(train_test_df[['tv']], train_test_df[['sales']], reg_tv)
    plt.show()

    plot_model(train_test_df[['radio']], train_test_df[['sales']], reg_radio)
    plt.show()

    plot_model(train_test_df[['newspaper']], train_test_df[['sales']], reg_newspaper)
    plt.show()

    rss_tv = calc_rss(validation_df[['sales']], reg_tv.predict(validation_df[['tv']]))
    rss_radio = calc_rss(validation_df[['sales']], reg_radio.predict(validation_df[['radio']]))
    rss_newspaper = calc_rss(validation_df[['sales']], reg_newspaper.predict(validation_df[['newspaper']]))

    print(f"RSS for tv-based model: {rss_tv}")
    print(f"RSS for radio-based model: {rss_radio}")
    print(f"RSS for newspaper-based model: {rss_newspaper}")

    r2_tv = calc_r2(validation_df[['sales']], reg_tv.predict(validation_df[['tv']]))
    r2_radio = calc_r2(validation_df[['sales']], reg_radio.predict(validation_df[['radio']]))
    r2_newspaper = calc_r2(validation_df[['sales']], reg_newspaper.predict(validation_df[['newspaper']])) 

    print(f"R2 for tv-based model: {r2_tv}")
    print(f"R2 for radio-based model: {r2_radio}")
    print(f"R2 for newspaper-based model: {r2_newspaper}")

