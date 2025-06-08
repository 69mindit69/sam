Importnumpyasnp
importpandasaspd
importmatplotlib.pyplotasplt
fromsklearn.datasetsimportload_breast_cancer
from sklearn.cluster import KMeans
fromsklearn.preprocessingimportStandardScaler from
sklearn.decomposition import PCA
importseabornassns
data= load_breast_cancer()
X = data.data
y=data.target
scaler =
StandardScaler()X_scaled=scaler.f
it_transform(X)
kmeans=KMeans(n_clusters=2,random_state=42)
kmeans.fit(X_scaled)
y_kmeans=kmeans.predict(X_scaled)
pca = PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:,0],y=X_pca[:,1],hue=y_kmeans,palette='viridis',s=100, alpha=0.7,
edgecolor='k')
centroids=pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:,0],centroids[:,1],s=300,c='red',marker='X',label='Centroids')
plt.title('K-means Clustering on Wisconsin Breast Cancer Dataset (PCA-reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('PrincipalComponent2')
plt.legend()
plt.show()