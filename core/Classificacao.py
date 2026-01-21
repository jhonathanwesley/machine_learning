# %%
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import linear_model, tree, naive_bayes
from pathlib import Path


data = Path('../data')
df = pd.read_excel(data/'dados_cerveja_nota.xlsx')
df
# %%
df['aprovado'] = (df['nota'] > 5).astype(int)
df
# %%
plt.figure(dpi=400)
plt.plot(df[['cerveja']], df['aprovado'], 'o', color="royalblue")

plt.title('Aprovações em Função de Cervejas')
plt.xlabel("Cervejas")
plt.ylabel("Aprovado")

plt.legend(['Aprovados & Ñ aprovados'])
plt.grid(True)
plt.show()
# %%
from sklearn import linear_model, tree, naive_bayes
reg_model = linear_model.LogisticRegression(
    penalty=None,
    fit_intercept=True,
)

features = ['cerveja']
target = 'aprovado'

X = df[features]
y = df[target]

reg_model.fit(X, y)

predict = reg_model.predict(X.drop_duplicates())
proba = reg_model.predict_proba(X.drop_duplicates())[:, 1]

arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(X, y)
arvore_full_predict = arvore_full.predict(X.drop_duplicates())
arvore_full_proba = arvore_full.predict_proba(X.drop_duplicates())[:, 1]

arvore_d2 = tree.DecisionTreeClassifier(max_depth=2, random_state=42)
arvore_d2.fit(X, y)
arvore_d2_predict = arvore_d2.predict(X.drop_duplicates())
arvore_d2_proba = arvore_d2.predict_proba(X.drop_duplicates())[:, 1]

nb = naive_bayes.GaussianNB()
nb.fit(X, y)
bayes_predict = nb.predict(X.drop_duplicates())
bayes_proba = nb.predict_proba(X.drop_duplicates())[:, 1]

plt.figure(dpi=400)
plt.plot(X, y, 'o', color="royalblue")

plt.plot(X.drop_duplicates(), predict, color='tomato')
plt.plot(X.drop_duplicates(), proba, color='red')

plt.plot(X.drop_duplicates(), arvore_full_predict, color='green')
plt.plot(X.drop_duplicates(), arvore_full_proba, color='purple')

plt.plot(X.drop_duplicates(), arvore_d2_predict, color='pink')
plt.plot(X.drop_duplicates(), arvore_d2_proba, color='cyan')

plt.plot(X.drop_duplicates(), bayes_predict, color='magenta')
plt.plot(X.drop_duplicates(), bayes_proba, color='grey')

plt.title('Aprovações em Função de Cervejas')
plt.xlabel("Cervejas")
plt.ylabel("Aprovado")
plt.hlines(.5, xmin=1, xmax=9, linestyles='--', colors='black')

plt.legend(['Observações', 'Regrss Predict', 'Regrss Prob', 'FullTree Predict', 'FullTree Prob', 'D2Tree Predict', 'D2Tree Prob', 'Bayes Predict', 'Bayes Prob'])
plt.grid(True)
plt.show()
