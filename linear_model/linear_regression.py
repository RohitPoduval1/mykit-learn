import numpy as np


class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D matrix. It currently is {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be a 1D vector. It currently is {y.ndim}D")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Shape mismatch between X and y "
                f"(X has {X.shape[0]} rows, y has {y.shape[0]} elements)"
            )

        # add ones (column vector of ones) for the bias/intercept term
        ones = np.ones((X.shape[0], 1))

        # put ones on the left for easy indexing (first weight is the bias)
        X_aug = np.hstack([ones, X])

        parameters = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
        self.intercept_ = parameters[0]
        self.coef_ = parameters[1:]


    def predict(self, X) -> np.ndarray:
        """
        Given a 2D matrix of examples and features (`n` examples and `d` features),
        return a `n` dimensional prediction vector (prediction for each example)
        """
        preds = X @ self.coef_ + self.intercept_
        if preds.shape[0] != X.shape[0]:
            raise ValueError(
                f"Different number of predictions compared to examples "
                f"({preds.shape[0]} vs. {X.shape[0]} respectively)"
            )

        return preds
