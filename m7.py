importpandasaspd
import numpy as np
importmatplotlib.pyplotasplt
fromsklearn.model_selectionimporttrain_test_split
fromsklearn.treeimportDecisionTreeClassifier, plot_tree
fromsklearn.metricsimportaccuracy_score,precision_score,recall_score,f1_score from
sklearn.preprocessing import LabelEncoder
titanic_data=pd.read_csv(r'C:\Users\Smart User\Documents\titanic.csv')
titanic_data=titanic_data.dropna(subset=['Survived','Pclass','Sex','Age','Embarked'])
label_encoder=LabelEncoder()
titanic_data['Sex'] = label_encoder.fit_transform(titanic_data['Sex'])
titanic_data['Embarked']=label_encoder.fit_transform(titanic_data['Embarked'])
X = titanic_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y =titanic_data['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred=dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision=precision_score(y_test,y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy:{accuracy*100:.2f}%")
print(f"Precision:{precision*100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")
plt.figure(figsize=(15, 10))
plot_tree(dt_classifier,feature_names=X.columns,class_names=['NotSurvived','Survived'],
filled=True, rounded=True)
plt.title("DecisionTree-TitanicDataset")
plt.show()