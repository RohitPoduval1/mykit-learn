import numpy as np
import matplotlib.pyplot as plt

from knn import KNNBase


class KNNRegression(KNNBase):
    def __init__(self, k: int, use_weighting: bool) -> None:
        super().__init__(k, use_weighting)

    def _predict_single(self, x):
        assert len(x.shape) == 1
        # distance from `x` to each point in the training data
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        assert len(distances.shape) == 1
        assert distances.shape[0] == self.X_train.shape[0]

        top_k_indices = distances.argsort()[:self.k]
        top_k_values = self.y_train[top_k_indices]

        # NOTE: Rather than a majority vote, we take the average of the
        # values of the neighbors for regression
        return np.mean(top_k_values)


def test_1d_simple():
    """Test case 1: Simple 1D data - house size vs price"""
    print("=" * 50)
    print("Test 1: House prices (1D)")
    print("=" * 50)
    
    # Training data: [square_feet] -> price (in $1000s)
    X_train = np.array([[1000], [1200], [1500], [1800], [2000], [2200]])
    y_train = np.array([150, 180, 225, 270, 300, 330])
    
    knn = KNNRegression(k=3, use_weighting=False)
    knn.fit(X_train, y_train)
    
    # Test predictions
    X_test = np.array([[1100], [1600], [2100]])
    predictions = knn.predict(X_test)
    
    print(f"Test points: {X_test.flatten()}")
    print(f"Predictions: {predictions}")
    print(f"Expected: ~165, ~247.5, ~315 (manual calculation)")
    print()
    
    # Visualize
    plt.figure(figsize=(10, 4))
    plt.scatter(X_train, y_train, c='blue', s=100, label='Training data')
    plt.scatter(X_test, predictions, c='red', s=100, marker='X', label='Predictions')
    plt.xlabel('Square Feet')
    plt.ylabel('Price ($1000s)')
    plt.legend()
    plt.title('KNN Regression: House Prices (k=3)')
    plt.grid(True, alpha=0.3)
    plt.show()


def test_2d_temperature():
    """Test case 2: 2D data - location to temperature"""
    print("=" * 50)
    print("Test 2: Temperature prediction (2D)")
    print("=" * 50)
    
    # Training data: [latitude, longitude] -> temperature
    X_train = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],
        [5, 5], [5, 6], [6, 5], [6, 6]
    ])
    y_train = np.array([20, 22, 21, 23, 30, 32, 31, 33])  # temperatures
    
    knn = KNNRegression(k=4, use_weighting=False)
    knn.fit(X_train, y_train)
    
    # Test predictions
    X_test = np.array([[0.5, 0.5], [5.5, 5.5], [3, 3]])
    predictions = knn.predict(X_test)
    
    print(f"Test points:")
    for point, pred in zip(X_test, predictions):
        print(f"  {point} -> {pred:.2f}°C")
    print()
    
    # Visualize
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                         s=200, cmap='coolwarm', alpha=0.6, 
                         edgecolors='black', linewidths=2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, 
               s=300, cmap='coolwarm', marker='X', 
               edgecolors='black', linewidths=3)
    plt.colorbar(scatter, label='Temperature (°C)')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('KNN Regression: Temperature Prediction (k=4)')
    plt.grid(True, alpha=0.3)
    plt.show()


def test_sinusoidal():
    """Test case 3: Learning a sine wave"""
    print("=" * 50)
    print("Test 3: Sine wave approximation")
    print("=" * 50)
    
    # Training data: sample sine wave
    X_train = np.linspace(0, 2*np.pi, 20).reshape(-1, 1)
    y_train = np.sin(X_train).flatten()
    
    # Test on finer grid
    X_test = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
    
    # Try different k values
    plt.figure(figsize=(12, 4))
    
    for i, k in enumerate([1, 3, 5], 1):
        knn = KNNRegression(k=k, use_weighting=False)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        plt.subplot(1, 3, i)
        plt.plot(X_test, np.sin(X_test), 'g-', label='True sine', linewidth=2)
        plt.scatter(X_train, y_train, c='blue', s=50, label='Training')
        plt.plot(X_test, predictions, 'r-', label=f'KNN (k={k})', alpha=0.7)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'k={k}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Notice: k=1 overfits, k=5 smooths more")
    print()


def test_edge_cases():
    """Test case 4: Edge cases"""
    print("=" * 50)
    print("Test 4: Edge cases")
    print("=" * 50)
    
    # Case 1: k=1 (should return exact neighbor)
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([10, 20, 30, 40])
    
    knn = KNNRegression(k=1, use_weighting=False)
    knn.fit(X_train, y_train)
    
    # Predict exact training point
    pred = knn.predict(np.array([[2]]))
    print(f"k=1, predict at training point [2]: {pred[0]} (expected: 20.0)")
    
    # Case 2: k equals dataset size
    knn_all = KNNRegression(k=4, use_weighting=False)
    knn_all.fit(X_train, y_train)
    pred = knn_all.predict(np.array([[100]]))  # Far away point
    print(f"k=4 (all data), far point [100]: {pred[0]} (expected: 25.0 = mean of all)")
    
    print()


def main():
    test_1d_simple()
    test_2d_temperature()
    test_sinusoidal()
    test_edge_cases()


if __name__ == '__main__':
    main()
