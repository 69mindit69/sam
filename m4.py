importnumpyasnp
importpandasaspd
from sklearn.model_selection import train_test_split
fromsklearn.metricsimportaccuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
iris=pd.read_csv(r'C:\Users\SmartUser\Documents\iris.csv')
print(iris.head())
print(iris.columns)
X=iris.drop('species',axis=1).values y
= iris['species'].values
print(X[:5])
print(y [:5])
X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
defknn_model(X_train,X_test,y_train,y_test,k,weighted=False): if
weighted:
model=KNeighborsClassifier(n_neighbors=k,weights='distance')
else:
model=KNeighborsClassifier(n_neighbors=k,weights='uniform')
model.fit(X_train, y_train)
y_pred= model.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
f1=f1_score(y_test,y_pred,average='weighted')
return accuracy, f1
k_values=[1,3, 5]
results={'k':[],'Regulark-NNAccuracy':[],'Regulark-NNF1Score':[],'Weightedk-NN Accuracy': [],
'Weighted k-NN F1 Score': []}
fork in k_values:
reg_accuracy,reg_f1=knn_model(X_train,X_test,y_train,y_test,k,weighted=False)
weighted_accuracy, weighted_f1 = knn_model(X_train, X_test, y_train, y_test, k,
weighted=True)
results['k'].append(k)
results['Regular k-NN Accuracy'].append(reg_accuracy)
results['Regular k-NN F1 Score'].append(reg_f1)
results['Weightedk-NNAccuracy'].append(weighted_accuracy)
results['Weighted k-NN F1 Score'].append(weighted_f1)
results_df=pd.DataFrame(results)
print(results_df)