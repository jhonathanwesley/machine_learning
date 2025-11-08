# %%
from sklearn import linear_model, tree
import pandas as pd


df = pd.read_excel('../data/dados_cerveja_nota.xlsx')

X = df[['cerveja']]     # X é uma Matriz, é um vetor bidimensional, com linhas e colunas
y = df['nota']          # y é um vetor

# Modelo
reg = linear_model.LinearRegression()

# O aprendizado - Algoritmo aprendendo com os dados
reg.fit(X, y)

a, b = reg.intercept_, reg.coef_[0]

predict_reg = reg.predict(X.drop_duplicates())

# Árvore completamente ajustada aos dados, provavelmente com overfitting
arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X, y)

predict_arvore_full = arvore_full.predict(X.drop_duplicates())

arvore_d2 = tree.DecisionTreeRegressor(random_state=42, max_depth=2)
arvore_d2.fit(X, y)

predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())

arvore_d1 = tree.DecisionTreeRegressor(random_state=42, max_depth=1)
arvore_d1.fit(X, y)

predict_arvore_d1 = arvore_d1.predict(X.drop_duplicates())

import matplotlib.pyplot as plt

# Plot relacionando variáveis x e y
plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title("Relação Entre Nota e Cervejas")
plt.xlabel("Cerveja")
plt.ylabel('Nota')

# Plot da regressão Linear: Reta passando pelos pontos de variáveis de entrada X e previsão y
plt.plot(X.drop_duplicates()['cerveja'], predict_reg)

# Plot das Regressões com Árvores de Decisão
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full)   # Sem limite de profundidade
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_d2) # Profundidade max é 2
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_d1) # Profundidade max é 3

# Legenda para cada elemento do gráfico
plt.legend(['Observado', f"f(x) = {a:.3f} + {b:.3f} x", 'Árvore Full', 'Árvore Depth = 2', 'Árvore Depth = 1'])
