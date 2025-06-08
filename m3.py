importpandasaspd
importmatplotlib.pyplotasplt
fromsklearn.decompositionimportPCA
from sklearn.datasets import load_iris
iris= load_iris()
iris_data=pd.DataFrame(iris.data,columns=iris.feature_names)
print(iris_data.describe())
print(iris_data.head())
X = iris.data
y=iris.target
plt.figure(figsize=(8, 6))
plt.scatter(X[:,0],X[:,1],c=y,cmap='viridis') pca
= PCA(n_components=2)
X_pca=pca.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:,0],X_pca[:,1],c=y,cmap='magma')
plt.show()