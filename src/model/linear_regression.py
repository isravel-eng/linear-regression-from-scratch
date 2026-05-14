"""Linear Regression model implementation from scratch."""

import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight = 0
        self.bias = 0
        self.loss_history = []

    def predict(self, X):
        return self.weight * X + self.bias

    def fit(self, X, y):
        n = len(X)

        for _ in range(self.epochs):
            y_pred = self.predict(X)

            dw = (-2 / n) * np.sum(X * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)

            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)
