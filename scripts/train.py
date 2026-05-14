"""Training script for Linear Regression."""

import numpy as np

from src.model.linear_regression import LinearRegression


X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X, y)

print(f"Weight: {model.weight}")
print(f"Bias: {model.bias}")
