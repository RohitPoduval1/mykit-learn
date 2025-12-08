import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

        # NOTE: Take a majority vote of the labels of the neighbors
        # and that majority will be the label of the new data point
        values, counts = np.unique(top_k_labels, return_counts=True)
        majority_vote = values[np.argmax(counts)]
        return majority_vote

def main():
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Without Scaling")
    for k in range(1, 36, 5):
        knn = KNNClassifier(k, use_weighting=False)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        accuracy = (preds == y_test).sum() / y_test.shape[0]
        print(f'\tAccuracy for k={k}: {accuracy:.3f}')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("With Scaling")
    for k in range(1, 36, 5):
        knn = KNNClassifier(k, use_weighting=False)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        accuracy = (preds == y_test).sum() / y_test.shape[0]
        print(f'\tAccuracy for k={k}: {accuracy:.3f}')

if __name__ == '__main__':
    main()
