#%%
from pathlib import Path
import pandas as pd


path = Path()
data = path/'../data'
# %%
'Os dados são a minha amostra, os exemplos, tabela é a abstração de algo do mundo real, nesse caso frutas.'
df = pd.read_excel(data/'dados_frutas.xlsx')
df
# %%
from sklearn import tree


arvore = tree.DecisionTreeClassifier(random_state=42)
# %%
y = df['Fruta']
caracteristicas = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']

X = df[caracteristicas]
arvore.fit(X, y)

arvore.predict([[1,1,1,1]])
# %%
arvore.predict([[0,1,1,1]])
# %%
arvore.predict([[1,0,1,1]])
# %%
arvore.predict([[1,1,0,1]])
# %%
arvore.predict([[1,1,1,0]])
# %%
import matplotlib.pyplot as plt


plt.figure(dpi=500)

tree.plot_tree(
    arvore,
    feature_names=caracteristicas,
    class_names=arvore.classes_,
    filled=True,
)
plt.show()
# %%
proba = arvore.predict_proba([[0, 0, 0, 0]])[0]
pd.Series(proba, index=arvore.classes_)
# %%
