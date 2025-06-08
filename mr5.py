import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)
def predict(x0):
w = np.exp(-(x - x0)**2 / (2 * 0.5**2))
W = np.diag(w)
X = np.c_[np.ones(len(x)), x]
theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
return np.array([1, x0]) @ theta
# Compute predictions
y_pred = np.array([predict(x0) for x0 in x])
# Plot
plt.scatter(x, y, c='r')
plt.plot(x, y_pred, c='b')
plt.show()