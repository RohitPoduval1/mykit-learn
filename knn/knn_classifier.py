import numpy as np
import matplotlib.pyplot as plt

from knn import KNNBase


class KNNClassifier(KNNBase):
    def __init__(self, k: int, use_weighting: bool) -> None:
        super().__init__(k, use_weighting)

    def _predict_single(self, x):
        assert len(x.shape) == 1
        # distance from `x` to each point in the training data
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        assert len(distances.shape) == 1
        assert distances.shape[0] == self.X_train.shape[0]

        top_k_indices = distances.argsort()[:self.k]
        top_k_labels = self.y_train[top_k_indices]

        majority_vote = np.argmax(np.bincount(top_k_labels))
        return majority_vote
