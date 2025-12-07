import numpy as np
import matplotlib.pyplot as plt


class KNNClassifier:
    def __init__(self, k: int) -> None:
        self.k = k

    def fit(self, X, y):
        # NOTE: No real training to be done because KNN is a lazy learner
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray):
        assert len(X.shape) == 2, "X must be a 2D array"
        predictions = []
        for x in X:
            predicted_label = self._predict_single(x)
            predictions.append(predicted_label)
        return np.array(predictions)

    def _predict_single(self, x: np.ndarray):
        # Calculate distance of point `x` to all training points
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))

        # Get k nearest neighbor indices
        top_k_smallest_distances = distances.argsort()[:self.k]

        # Get their labels
        top_k_labels = self.y_train[top_k_smallest_distances]

        # Majority vote
        majority_vote = np.argmax(np.bincount(top_k_labels))
        return majority_vote


def main():
    X_train = np.array([
        [1, 1], [1, 2], [2, 1], [2, 2],
        [-1, -1], [-1, -2], [-2, -1], [-2, -2]
    ])
    y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    plt.scatter(X_train[:, 0], X_train[:, 1], c=np.array([{0: 'red', 1: 'blue'}[label] for label in y_train]))
    plt.show()

    knn = KNNClassifier(4)
    knn.fit(X_train, y_train)
    arr = np.array([1.5, 1.5])
    prediction = knn._predict_single(arr)
    print(f"Prediction for {arr}: {prediction}")

    arr = np.array( [0, 0] )
    prediction = knn._predict_single(arr)
    print(f"Prediction for {arr}: {prediction}")

    arr = np.array([0.2, 0.2])
    prediction = knn._predict_single(arr)
    print(f"Prediction for {arr}: {prediction}")

if __name__ == '__main__':
    main()
