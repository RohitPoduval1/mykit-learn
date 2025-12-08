import numpy as np
import matplotlib.pyplot as plt


class KNNBase:
    def __init__(self, k: int, use_weighting: bool) -> None:
        self.k = k
        self.use_weighting = use_weighting 

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray):
        assert len(X.shape) == 2, "X must be a 2D array"
        predictions = np.array([self._predict_single(x) for x in X])
        return predictions

    def _predict_single(self, x):
        raise NotImplementedError()
