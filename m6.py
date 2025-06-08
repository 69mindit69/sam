importnumpyasnp
importpandasaspd
importmatplotlib.pyplotasplt
fromsklearn.model_selectionimporttrain_test_split
from sklearn.linear_model import LinearRegression
fromsklearn.preprocessingimportPolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
fromsklearn.metricsimportmean_squared_error,r2_score from
sklearn.datasets import fetch_openml
#Part1:LinearRegressionwithBostonHousingDataset #
Load Boston Housing Dataset
boston=fetch_openml(name='boston', version=1)
X_boston=pd.DataFrame(boston.data,columns=boston.feature_names) y_boston
= boston.target.astype(float)
# Split data
X_train_b,X_test_b,y_train_b,y_test_b=train_test_split( X_boston,
y_boston, test_size=0.2, random_state=42
)
#Trainand evaluateLinear Regression
lr = LinearRegression()
lr.fit(X_train_b,y_train_b)
#ConvertallcolumnsinX_test_btonumeric,coercingerrors
#Thisstepensuresallfeaturesareinaformatsuitablefornumericaloperations for col in
X_test_b.columns:
X_test_b[col]=pd.to_numeric(X_test_b[col],errors='coerce')
#Optional:HandlepotentialNaNsintroducedbycoercionifanynon-numericvaluesexisted # If the
dataset is clean from fetch_openml, this might not be strictly necessary,
# but it's a safeguard.
#X_test_b= X_test_b.fillna(X_test_b.mean())# Example:fillNaNs withcolumn mean
#Debugging:CheckdatatypesofX_test_bafterconversion print("Data
types of X_test_b after numeric conversion:")
print(X_test_b.dtypes)
y_pred_b= lr.predict(X_test_b)
print("Boston Housing - Linear Regression Results:")
print(f"MSE:{mean_squared_error(y_test_b,y_pred_b):.2f}")
print(f"R²: {r2_score(y_test_b, y_pred_b):.2f}\n")
# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(y_test_b,y_pred_b,alpha=0.5)
plt.plot([y_test_b.min(), y_test_b.max()],
[y_test_b.min(),y_test_b.max()],'k--',lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('PredictedPrices')
plt.title('BostonHousing:ActualvsPredictedPrices')
plt.show()
#Part2:PolynomialRegressionwithAutoMPGDataset #
Load and preprocess Auto MPG data
url="https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ['mpg', 'cylinders', 'displacement', 'horsepower',
'weight','acceleration','model_year','origin','name']
auto_df=pd.read_csv(url,delim_whitespace=True,header=None,names=columns)
#Cleandata
auto_df['horsepower']=pd.to_numeric(auto_df['horsepower'],errors='coerce')
auto_df = auto_df.dropna().reset_index(drop=True)
auto_df=auto_df.drop('name', axis=1)
X_auto=auto_df.drop('mpg',axis=1)
y_auto = auto_df['mpg']
# Split data
X_train_a,X_test_a,y_train_a,y_test_a=train_test_split(
X_auto, y_auto, test_size=0.2, random_state=42
)
#CreatePolynomialRegressionpipeline
degree = 2
poly_reg= Pipeline([
('poly',PolynomialFeatures(degree=degree)),
('scaler', StandardScaler()),
('regressor',LinearRegression())
])
# Train and evaluate
poly_reg.fit(X_train_a, y_train_a)
y_pred_a=poly_reg.predict(X_test_a)
print("AutoMPG-PolynomialRegressionResults:") print(f"Degree:
{degree}")
print(f"MSE:{mean_squared_error(y_test_a,y_pred_a):.2f}")
print(f"R²: {r2_score(y_test_a, y_pred_a):.2f}")
# Plot results
plt.figure(figsize=(8,6))
plt.scatter(y_test_a,y_pred_a,alpha=0.5)
plt.plot([y_test_a.min(), y_test_a.max()],
[y_test_a.min(),y_test_a.max()],'k--',lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('PredictedMPG')
plt.title('AutoMPG:ActualvsPredictedFuelEfficiency')
plt.show()