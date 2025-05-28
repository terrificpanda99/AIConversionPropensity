import math

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0.0

    def fit(self, X, y):
        if not X:
            return
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                linear = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                pred = 1.0 / (1.0 + math.exp(-linear))
                error = pred - yi
                for j in range(n_features):
                    self.weights[j] -= self.learning_rate * error * xi[j]
                self.bias -= self.learning_rate * error

    def predict_proba(self, X):
        return [self._predict_single(xi) for xi in X]

    def predict_single(self, xi):
        return self._predict_single(xi)

    def _predict_single(self, xi):
        linear = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
        return 1.0 / (1.0 + math.exp(-linear))
