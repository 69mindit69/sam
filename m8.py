fromsklearnimportdatasets
fromsklearn.model_selectionimporttrain_test_split
from sklearn.naive_bayes import GaussianNB
fromsklearn.metricsimportaccuracy_score
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X = iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred=nb_classifier.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print(f'AccuracyofNaiveBayesclassifier:{accuracy*100:.2f}%')
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:,0],X_test[:,1],c=y_pred,cmap='coolwarm',marker='o',edgecolor='k')
plt.title('Naive Bayes Classifier - Test Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.colorbar(label='PredictedSpecies')
plt.show()