#%%
# Model Com Dataset de Star Wars
from functools import cache
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

path = Path()
data = path/'../data'

parquet_path = data/'dados_clones.parquet'
    
try:
    df = pd.read_parquet(parquet_path)

except Exception as e:

    try:
        df = pd.read_parquet(parquet_path, engine='fastparquet')

    except Exception as ex:
        raise RuntimeError(f'ERRO: {ex}.\n\n==> Verifique as versões instaladas de pandas/pyarrow e considere reinstalar/atualizar. ')

df.rename(columns={'Status ': 'Status'}, inplace=True)
df
# %%
df.dtypes
# %%
df.shape
# %%
from sklearn import tree


sw_model = tree.DecisionTreeClassifier()

learn_df = df.replace({
    'Tipo 1': 1, 'Tipo 2': 2, 'Tipo 3': 3, 'Tipo 4': 4, 'Tipo 5': 5,
    'Defeituoso': 0, 'Apto': 1,
    #'Yoda': 1, 'Shaak Ti': 1, 'Obi-Wan Kenobi': 3, 'Aayla Secura': 4, 'Mace Windu': 5,
    })

sw_features = [
    "Massa(em kilos)", "Estatura(cm)",#    "Distância Ombro a ombro", "Tamanho do crânio", "Tamanho dos pés", "Tempo de existência(em meses)"
    ]
sw_target = "Status"

sw_X = learn_df[sw_features]
sw_y = learn_df[sw_target]

sw_model.fit(X=sw_X, y=sw_y)
# %%
import matplotlib.pyplot as plt


plt.figure(dpi=500)

sw_class_names = [str(class_item) for class_item in sw_model.classes_]

tree.plot_tree(
    sw_model,
    feature_names=sw_features,
    class_names=sw_class_names,
    filled=True,
    max_depth=3,
)
# %%
