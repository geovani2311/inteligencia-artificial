
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# =========================
# Função genérica de regressão com RSS e R²
# =========================
def regressao_sgd(df, x_columns, y_column, max_iter=5000, plot_graph=True):
    X = df[x_columns].values
    y = df[y_column].values.ravel()

    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Regressão com SGD
    reg_grad = SGDRegressor(max_iter=max_iter, alpha=0, random_state=42)
    reg_grad.fit(X_scaled, y)

    y_pred = reg_grad.predict(X_scaled)

    # Métricas de validação
    rss = np.sum((y - y_pred) ** 2)
    r2 = r2_score(y, y_pred)

    print(f"=== Regressão SGD para {x_columns} → '{y_column}' ===")
    print(f"Intercepto: {reg_grad.intercept_[0]:.5f}")
    print(f"Coeficientes: {reg_grad.coef_}")
    print(f"RSS: {rss:.2f}")
    print(f"R²: {r2:.5f}")
    print()

    if plot_graph and len(x_columns) == 1:
        # Reverter normalização para plotar linha sobre X original
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_plot_scaled = scaler.transform(X_plot)
        y_plot_pred = reg_grad.predict(X_plot_scaled)

        plt.scatter(X, y, color='g', label='Dados reais')
        plt.plot(X_plot, y_plot_pred, color='k', label='Regressão')
        plt.xlabel(x_columns[0])
        plt.ylabel(y_column)
        plt.title(f'{x_columns[0]} → {y_column}')
        plt.legend()
        plt.grid(True)
        plt.show()

# =========================
# Carregar datasets
# =========================
df_advertising = pd.read_csv('data/Advertising.csv', header=0, names=['tv', 'radio', 'newspaper', 'sales'])
df_portland = pd.read_csv('data/Portland_housePrices.csv')

# =========================
# Aplicar regressão SGD para atributos desejados
# =========================
regressao_sgd(df_advertising, ['tv'], 'sales', plot_graph=False)
regressao_sgd(df_advertising, ['radio'], 'sales', plot_graph=False)
regressao_sgd(df_advertising, ['newspaper'], 'sales', plot_graph=False)
regressao_sgd(df_portland, ['tamanho'], 'preco', plot_graph=False)
regressao_sgd(df_portland, ['quartos'], 'preco', plot_graph=False)

# =========================
# Aplicar regressão SGD para todos os atributos
# =========================
regressao_sgd(df_advertising, ['tv', 'radio', 'newspaper'], 'sales', plot_graph=False)
regressao_sgd(df_portland, ['tamanho', 'quartos'], 'preco', plot_graph=False)
