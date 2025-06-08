from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
result = PCA(n_components=2).fit_transform(iris.data)
plt.scatter(result[:,0], result[:,1], c=iris.target)
plt.title("PCA - Iris")
plt.show()