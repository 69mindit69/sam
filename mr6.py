import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
X, y = fetch_california_housing(return_X_y=True)
model = LinearRegression().fit(X[:, [0]], y)
plt.scatter(X[:, 0], y, s=1)
plt.plot(X[:, 0], model.predict(X[:, [0]]), c='r')
plt.title("Linear Regression")
plt.show()
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ['mpg','cyl','disp','hp','wt','acc','yr','ori','name']
df = pd.read_csv(url, names=columns, sep=r'\s+', na_values='?').dropna() # updated sep
Xv2 = df[['hp']].astype(float).values
yv2 = df['mpg'].values
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(Xv2, yv2)
yp2 = poly_model.predict(Xv2)
plt.scatter(Xv2, yv2, c='b', s=10, label='Actual')
plt.scatter(Xv2, yp2, c='r', s=10, label='Predicted')
plt.title("Polynomial Regression")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.show()