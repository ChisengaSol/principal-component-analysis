import pandas as pd
from pca import PCA
from sklearn.datasets import load_iris

iris = load_iris()
X = iris['data']
y = iris['target']

n_samples, n_features = X.shape

print('Number of samples:', n_samples)
print('Number of features:', n_features)

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
    )
df["label"] = iris.target

df.info()

if __name__ == '__main__':
    my_pca = PCA(n_component=2)
    my_pca.fit(X)
    new_X = my_pca.transform(X)
    print(new_X)