import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import numpy as np
from sklearn.linear_model import LinearRegression

df_advertising = pd.read_csv('data/Advertising.csv', header=0, names=['tv', 'radio', 'newspaper', 'sales'])
df_portland = pd.read_csv('data/Portland_housePrices.csv', index_col=None)

########################################################
# Dividir os dados
########################################################

def data_splitting(df, x_column, y_column):
    X_train, X_val, y_train, y_val = train_test_split(df[x_column], df[y_column[0]], test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

########################################################
# Criar e treinar o modelo KNN
########################################################
def knn_model(X_train, y_train):
    knn = KNeighborsRegressor(n_neighbors=5)
    return knn.fit(X_train, y_train)

########################################################
# Criar e treinar o modelo Linear Regression
########################################################
def linear_regression_model(X_train, y_train):
    reg = LinearRegression().fit(X_train, y_train)
    return reg

########################################################
# Avaliação (treino e validação)
########################################################

def evaluate_model(model_name, model, X_train, y_train, X_val, y_val):
    # Previsão no treino
    y_train_pred = model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    rss_train = np.sum((np.array(y_train) - np.array(y_train_pred)) ** 2)

    # Previsão na validação
    y_val_pred = model.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)
    rss_val = np.sum((np.array(y_val) - np.array(y_val_pred)) ** 2)

    print(f"{model_name} - Conjunto de Treinamento:")
    print(f"  R²:  {r2_train:.4f}")
    print(f"  RSS: {rss_train:.2f}")

    print(f"{model_name} - Conjunto de Validação:")
    print(f"  R²:  {r2_val:.4f}")
    print(f"  RSS: {rss_val:.2f}\n")

if __name__ == "__main__":
    X_train_advertising, X_val_advertising, y_train_advertising, y_val_advertising = data_splitting(df_advertising, ['tv', 'radio', 'newspaper'], ['sales'])
    X_train_portland, X_val_portland, y_train_portland, y_val_portland = data_splitting(df_portland, ['tamanho', 'quartos'], ['preco'])

    ########################################################
    # Treinamento dos modelos
    ########################################################
    knn_advertising = knn_model(X_train_advertising, y_train_advertising)
    knn_portland = knn_model(X_train_portland, y_train_portland)

    linear_regression_advertising = linear_regression_model(X_train_advertising, y_train_advertising)
    linear_regression_portland = linear_regression_model(X_train_portland, y_train_portland)    

    ########################################################
    # Treinamento dos modelos no data split
    ########################################################
    knn_advertising_no_data_split = knn_model(df_advertising[['tv', 'radio', 'newspaper']], df_advertising['sales'])
    linear_regression_advertising_no_data_split = linear_regression_model(df_advertising[['tv', 'radio', 'newspaper']], df_advertising['sales'])
    knn_portland_no_data_split = knn_model(df_portland[['tamanho', 'quartos']], df_portland['preco'])
    linear_regression_portland_no_data_split = linear_regression_model(df_portland[['tamanho', 'quartos']], df_portland['preco'])

    ########################################################
    # Avaliação dos modelos
    ########################################################
    evaluate_model('KNN Advertising', knn_advertising, X_train_advertising, y_train_advertising, X_val_advertising, y_val_advertising)
    evaluate_model('KNN Portland', knn_portland, X_train_portland, y_train_portland, X_val_portland, y_val_portland)
    evaluate_model('Linear Regression Advertising', linear_regression_advertising, X_train_advertising, y_train_advertising, X_val_advertising, y_val_advertising)
    evaluate_model('Linear Regression Portland', linear_regression_portland, X_train_portland, y_train_portland, X_val_portland, y_val_portland)
    evaluate_model('KNN Advertising no data split', knn_advertising_no_data_split, df_advertising[['tv', 'radio', 'newspaper']], df_advertising['sales'], df_advertising[['tv', 'radio', 'newspaper']], df_advertising['sales'])
    evaluate_model('Linear Regression Advertising no data split', linear_regression_advertising_no_data_split, df_advertising[['tv', 'radio', 'newspaper']], df_advertising['sales'], df_advertising[['tv', 'radio', 'newspaper']], df_advertising['sales'])
    evaluate_model('KNN Portland no data split', knn_portland_no_data_split, df_portland[['tamanho', 'quartos']], df_portland['preco'], df_portland[['tamanho', 'quartos']], df_portland['preco'])
    evaluate_model('Linear Regression Portland no data split', linear_regression_portland_no_data_split, df_portland[['tamanho', 'quartos']], df_portland['preco'], df_portland[['tamanho', 'quartos']], df_portland['preco'])

    print("=" * 60)
    print("Resultados de Cross-Validation (R² médio)")
    print("=" * 60)

    cv_scores_advertising = cross_val_score(linear_regression_advertising, X_train_advertising, y_train_advertising, cv=5, scoring='r2')
    cv_scores_portland = cross_val_score(linear_regression_portland, X_train_portland, y_train_portland, cv=5, scoring='r2')
    cv_scores_knn_advertising = cross_val_score(knn_advertising, X_train_advertising, y_train_advertising, cv=5, scoring='r2')
    cv_scores_knn_portland = cross_val_score(knn_portland, X_train_portland, y_train_portland, cv=5, scoring='r2')
    cv_scores_knn_advertising_no_data_split = cross_val_score(knn_advertising_no_data_split, df_advertising[['tv', 'radio', 'newspaper']], df_advertising['sales'], cv=5, scoring='r2')
    cv_scores_knn_portland_no_data_split = cross_val_score(knn_portland_no_data_split, df_portland[['tamanho', 'quartos']], df_portland['preco'], cv=5, scoring='r2')

    print(f"R² médio CV - Linear Regression Advertising: {cv_scores_advertising.mean():.4f}")
    print(f"R² médio CV - Linear Regression Portland:    {cv_scores_portland.mean():.4f}")
    print(f"R² médio CV - KNN Advertising:                {cv_scores_knn_advertising.mean():.4f}")
    print(f"R² médio CV - KNN Portland:                   {cv_scores_knn_portland.mean():.4f}")
    print(f"R² médio CV - KNN Advertising no data split:  {cv_scores_knn_advertising_no_data_split.mean():.4f}")
    print(f"R² médio CV - KNN Portland no data split:     {cv_scores_knn_portland_no_data_split.mean():.4f}")