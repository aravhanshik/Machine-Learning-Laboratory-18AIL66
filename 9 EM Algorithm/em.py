import pandas as pd
from sklearn.mixture import GaussianMixture

df = pd.read_csv('iris.csv')
X = df.iloc[:, :-1]
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
labels = gmm.predict(X)
df['Cluster'] = labels
print(df)
