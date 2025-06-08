import numpy as np
import pandas as pd
importseabornassns
importmatplotlib.pyplotasplt
fromsklearn.datasetsimport load_iris
iris= load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names) x
= df['sepal length (cm)']
y=df['petallength (cm)']
plt.figure(figsize=(8, 6))
plt.scatter(x,y,color='blue',alpha=0.6)
plt.title('Scatter Plot')
plt.xlabel('SepalLength(cm)')
plt.ylabel('Petal Length (cm)')
plt.show()
corr=np.corrcoef(x,y)[0, 1]
print(f"Pearsoncorrelationcoefficientis: {corr:.2f}")
cov_mat = df.cov()
print("\nCovarianceMatrix:")
print(cov_mat)
corr_mat = df.corr()
print("\nCorrelationMatrix:")
print(corr_mat)
plt.figure(figsize=(8, 6))
sns.heatmap(corr_mat,annot=True,cmap='coolwarm',fmt='.2f',cbar=True,linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()