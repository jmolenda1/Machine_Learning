import numpy as np

from sklearn.datasets import fetch_california_housing

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

california = fetch_california_housing() # Bunch object

print(california.DESCR)

print(california.data.shape)

print(california.target.shape)

print(california.feature_names)

california_df = pd.DataFrame(california.data, columns=california.feature_names)

california_df["MedHouseValue"] = pd.Series(california.target)

sns.set(font_scale=2)
sns.set_style("whitegrid")

grid = sns.pairplot(data=california_df, vars=california_df.columns[0:])

plt.show()




