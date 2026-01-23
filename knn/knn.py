import numpy as np


class KNNBase:
    def __init__(self, k: int, use_weighting: bool) -> None:
        self.k = k
        self.use_weighting = use_weighting


    def fit(self, X, y):
        """
        Because KNN is a lazy learner, there are no parameters to learn
        from the data, so all fitting does is just store the training data.
        """
        self.X_train = X
        self.y_train = y


    # NOTE: KNN for Classification and Regression are very similar except
    # for what is done with the top k targets. That is what subclasses implement
    def _get_top_k_targets(self, x) -> np.ndarray:
        """
        Given a feature vector (single data point) `x` return the targets
        (labels in classification, values in regression) of the top k closest
        data points to the incoming point

        Args:
            x (np.array) - d dimensional vector

        Returns:
            k dimensional vector containing targets of the top k closest data
            points in the training set sorted in descending order, meaning
            result[0] is the target of the closest point
        """
        assert len(x.shape) == 1, "x must be a *vector* of points"

        # squared distance from `x` to each point in the training data
        distances = np.sum((self.X_train - x) ** 2, axis=1)
        assert len(distances.shape) == 1
        assert distances.shape[0] == self.X_train.shape[0], (
            "There must be the same number of distances as there are points "
            "in the training dataset"
        )

        top_k_indices = distances.argsort()[:self.k]
        top_k_targets = self.y_train[top_k_indices]

        return top_k_targets


    def predict(self, X: np.ndarray):
        """
        Given a matrix of feature vectors, return a vector of predictions
        """
        assert len(X.shape) == 2, "X must be a 2D array"
        predictions = np.array([self._predict_single(x) for x in X])
        return predictions


    def _predict_single(self, x):
        """Return the prediction for a single feature vector

        Args:
            x (np.array) - d dimensional vector
        """
        raise NotImplementedError()
