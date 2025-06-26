
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt

# =========================
# Função genérica de regressão
# =========================
def regressao_sgd(df, x_column, y_column, max_iter=5000, plot_graph=True):
    X = df[[x_column]]
    y = df[[y_column]]

    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Regressão com SGD
    reg_grad = SGDRegressor(max_iter=max_iter, alpha=0)
    reg_grad.fit(X_scaled, y.values.ravel())

    print(f"=== Regressão SGD para '{x_column}' → '{y_column}' ===")
    print(f"Intercepto: {reg_grad.intercept_[0]:.5f}")
    print(f"Coeficiente: {reg_grad.coef_[0]:.5f}")
    print()

    if plot_graph:
        # Reverter normalização para plotar linha sobre X original
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_plot_scaled = scaler.transform(X_plot)
        y_pred = reg_grad.predict(X_plot_scaled)

        plt.scatter(X, y, color='g', label='Dados reais')
        plt.plot(X_plot, y_pred, color='k', label='Regressão')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f'{x_column} → {y_column}')
        plt.legend()
        plt.grid(True)
        plt.show()
    return reg_grad
# =========================
# Carregar datasets
# =========================
df_advertising = pd.read_csv('data/Advertising.csv', header=0, names=['tv', 'radio', 'newspaper', 'sales'])
df_portland = pd.read_csv('data/Portland_housePrices.csv')

# =========================
# Aplicar regressão SGD para atributos desejados
# =========================
regressao_sgd(df_advertising, 'tv', 'sales', plot_graph=False)
regressao_sgd(df_advertising, 'tv', 'sales', plot_graph=False)
regressao_sgd(df_advertising, 'tv', 'sales', plot_graph=False)
regressao_sgd(df_portland, 'tamanho', 'preco', plot_graph=False)
regressao_sgd(df_portland, 'tamanho', 'preco', plot_graph=False)
regressao_sgd(df_portland, 'tamanho', 'preco', plot_graph=False)

