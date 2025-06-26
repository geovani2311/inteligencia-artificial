
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =========================
# Função genérica com gráfico de resíduos
# =========================
def regressao_sgd(df, x_columns, y_column, max_iter=5000, plot_graph=True, plot_residuos=True):
    X = df[x_columns].values
    y = df[y_column].values.ravel()

    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Regressão com SGD
    reg_grad = SGDRegressor(max_iter=max_iter, alpha=0, random_state=42)
    reg_grad.fit(X_scaled, y)

    y_pred = reg_grad.predict(X_scaled)

    # Gráfico de resíduos
    if plot_residuos:
        residuos = y - y_pred
        plt.figure()
        plt.scatter(y_pred, residuos, color='purple', alpha=0.6)
        plt.axhline(0, color='black', linestyle='--')
        plt.title('Gráfico de Resíduos')
        plt.xlabel('Valores Previstos')
        plt.ylabel('Resíduos')
        plt.grid(True)
        plt.show()

# =========================
# Carregar datasets
# =========================
def load_data():
    df_advertising = pd.read_csv('data/Advertising.csv', header=0, names=['tv', 'radio', 'newspaper', 'sales'])
    df_portland = pd.read_csv('data/Portland_housePrices.csv')
    return df_advertising, df_portland


def regressao_com_pvalor(df, x_columns, y_column, alpha=0.05):
    X = df[x_columns]
    y = df[y_column]

    # Adiciona constante para statsmodels
    X_const = sm.add_constant(X)

    # Ajusta modelo OLS
    ols_model = sm.OLS(y, X_const).fit()
    pvalores = ols_model.pvalues
    print("P-valores dos preditores:\n", pvalores)

    # Seleciona variáveis significativas
    significant = pvalores[pvalores < alpha].index.drop("const", errors="ignore")
    print("\nVariáveis significativas:", list(significant))

    if len(significant) == 0:
        print("Nenhuma variável significativa encontrada com p <", alpha)
        return

    # Reajuste com variáveis significativas
    X_sig = X[significant]
    lr_sig = LinearRegression()
    lr_sig.fit(X_sig, y)
    y_pred = lr_sig.predict(X_sig)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("\nDesempenho do modelo com variáveis significativas:")
    print("MSE:", mse)
    print("R²:", r2)

    # Gráfico de resíduos
    residuos = y - y_pred
    plt.figure()
    plt.scatter(y_pred, residuos, color='darkred', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Gráfico de Resíduos (Variáveis Significativas)')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Resíduos')
    plt.grid(True)
    plt.show()

# Calcula VIF para Advertising
def calc_vif(df, x_columns, y_column):        
    X_adv = df[x_columns]
    X_adv_const = sm.add_constant(X_adv)

    vif = pd.DataFrame()
    vif["Variável"] = X_adv_const.columns
    vif["VIF"] = [variance_inflation_factor(X_adv_const.values, i) for i in range(X_adv_const.shape[1])]

    print("=== VIF para ", y_column, " ===")
    print(vif)

# =========================
# Aplicar regressão SGD
# =========================
def main():
    df_advertising, df_portland = load_data()
    '''
    regressao_com_pvalor(df_advertising, ['tv', 'radio', 'newspaper'], 'sales')
    regressao_com_pvalor(df_portland, ['tamanho', 'quartos'], 'preco')
    regressao_sgd(df_advertising, ['tv', 'radio', 'newspaper'], 'sales', max_iter=1000)
    regressao_sgd(df_portland, ['tamanho', 'quartos'], 'preco', max_iter=1000)
    '''
    calc_vif(df_advertising, ['tv', 'radio', 'newspaper'], 'sales')
    calc_vif(df_portland, ['tamanho', 'quartos'], 'preco')


if __name__ == "__main__":
    main()