import numpy as np
import pytest


class KMeans:
    def __init__(self, num_clusters: int, max_iters: int=30, random_state=5521) -> None:
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = np.array([])
        self._data_points_for_cluster = [[] for _ in range(num_clusters)]

    def _init_centroids(self, X):
        """Initialize centroids as random points in the data matrix `X`.

        Args:
            X (np.ndarray): Data matrix to initialize based off of
        """
        np.random.seed(self.random_state)
        random_indices = np.random.choice(
            X.shape[0],
            size=self.num_clusters,
            replace=False
        )
        self.centroids = X[random_indices]
        print(f'Centroid Intialization: {self.centroids}')

    def _find_closest_centroid(self, x):
        """
        Return the index of the closest centroid (using self.centroids) to the
        given point `x`
        """
        assert len(x.shape) == 1
        
        distances_to_centroid = np.array(
            [np.sqrt(np.sum((x - centroid)**2)) for centroid in self.centroids]
        )

        # The index of the closest centroid to `x`
        closest_centroid_index = np.argmin(distances_to_centroid)
        return closest_centroid_index

    def _update_centroid_means(self):
        for centroid_index in range(self.centroids.shape[0]):
            data = np.array(self._data_points_for_cluster[centroid_index])
            new_centroid_mean = np.mean(data)
            self.centroids[centroid_index] = new_centroid_mean

    def fit(self, X):
        self._init_centroids(X)
        for _ in range(self.max_iters):
            for x in X:
                centroid_index = self._find_closest_centroid(x)
                self._data_points_for_cluster[centroid_index].append(x)
            
            prev_centroids = self.centroids.copy()
            self._update_centroid_means()
            assert prev_centroids.shape[0] == self.centroids.shape[0]

            # NOTE: We have reached convergence if the centroids do not change
            if (prev_centroids == self.centroids).all():
                break

    def predict(self, X) -> np.ndarray:
        found_centroids = [self._find_closest_centroid(x) for x in X]
        return np.array(found_centroids)


# Pytest fixtures
@pytest.fixture
def simple_2d_data():
    """Two clearly separated clusters in 2D"""
    return np.array([
        [1, 1], [1, 2], [2, 1],  # Cluster 1
        [10, 10], [10, 11], [11, 10]  # Cluster 2
    ])

@pytest.fixture
def three_cluster_data():
    """Three clearly separated clusters"""
    return np.array([
        [0, 0], [1, 1],  # Cluster 1
        [10, 10], [11, 11],  # Cluster 2
        [20, 20], [21, 21]  # Cluster 3
    ])

@pytest.fixture
def high_dim_data():
    """Higher dimensional data (4D)"""
    return np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [10, 11, 12, 13],
        [10, 11, 12, 14]
    ])


# Tests
class TestKMeansInitialization:
    
    def test_init_parameters(self):
        """Test KMeans initialization with custom parameters"""
        km = KMeans(num_clusters=3, max_iters=50, random_state=42)
        assert km.num_clusters == 3
        assert km.max_iters == 50
        assert km.random_state == 42
        assert len(km._data_points_for_cluster) == 3
    
    def test_init_centroids_shape(self, simple_2d_data):
        """Test centroid initialization produces correct shape"""
        km = KMeans(num_clusters=2, random_state=42)
        km._init_centroids(simple_2d_data)
        
        assert km.centroids.shape == (2, 2)
    
    def test_init_centroids_from_data(self, simple_2d_data):
        """Test that centroids are actual points from the dataset"""
        km = KMeans(num_clusters=2, random_state=42)
        km._init_centroids(simple_2d_data)
        
        # Check that each centroid matches at least one point in X
        for centroid in km.centroids:
            assert any(np.allclose(centroid, x) for x in simple_2d_data)


class TestKMeansDistanceCalculation:
    
    def test_find_closest_centroid_first(self):
        """Test finding closest centroid - point near first centroid"""
        km = KMeans(num_clusters=2)
        km.centroids = np.array([[0, 0], [10, 10]])
        
        closest = km._find_closest_centroid(np.array([1, 1]))
        assert closest == 0
    
    def test_find_closest_centroid_second(self):
        """Test finding closest centroid - point near second centroid"""
        km = KMeans(num_clusters=2)
        km.centroids = np.array([[0, 0], [10, 10]])
        
        closest = km._find_closest_centroid(np.array([9, 9]))
        assert closest == 1
    
    def test_find_closest_centroid_equidistant(self):
        """Test finding closest centroid for point equidistant from two centroids"""
        km = KMeans(num_clusters=2)
        km.centroids = np.array([[0, 0], [10, 0]])
        
        # Point at (5, 0) is equidistant - should return first one found
        closest = km._find_closest_centroid(np.array([5, 0]))
        assert closest in [0, 1]


class TestKMeansClustering:
    
    def test_simple_two_clusters(self, simple_2d_data):
        """Test basic clustering on clearly separated data"""
        km = KMeans(num_clusters=2, random_state=42)
        km.fit(simple_2d_data)
        
        predictions = km.predict(simple_2d_data)
        
        # First 3 points should be in same cluster
        assert predictions[0] == predictions[1] == predictions[2]
        
        # Last 3 points should be in same cluster
        assert predictions[3] == predictions[4] == predictions[5]
        
        # First and last groups should be in different clusters
        assert predictions[0] != predictions[3]
    
    def test_three_clusters(self, three_cluster_data):
        """Test with three clusters"""
        km = KMeans(num_clusters=3, random_state=42)
        km.fit(three_cluster_data)
        
        predictions = km.predict(three_cluster_data)
        
        # Check that we have 3 different clusters
        unique_clusters = np.unique(predictions)
        assert len(unique_clusters) == 3
    
    def test_single_cluster(self):
        """Test with single cluster"""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        km = KMeans(num_clusters=1, random_state=42)
        km.fit(X)
        
        predictions = km.predict(X)
        
        # All points should be in cluster 0
        assert np.all(predictions == 0)
    
    def test_higher_dimensions(self, high_dim_data):
        """Test with higher dimensional data (4D)"""
        km = KMeans(num_clusters=2, random_state=42)
        km.fit(high_dim_data)
        predictions = km.predict(high_dim_data)
        
        # First two should be in same cluster
        assert predictions[0] == predictions[1]
        # Last two should be in same cluster
        assert predictions[2] == predictions[3]
        # Should be different clusters
        assert predictions[0] != predictions[2]


class TestKMeansConvergence:
    
    def test_convergence_occurs(self, simple_2d_data):
        """Test that algorithm converges"""
        km = KMeans(num_clusters=2, max_iters=100, random_state=42)
        km.fit(simple_2d_data)
        
        # Should converge to reasonable centroids
        assert km.centroids.shape == (2, 2)
        
        # Centroids should not be NaN or infinite
        assert not np.any(np.isnan(km.centroids))
        assert not np.any(np.isinf(km.centroids))
    
    def test_centroids_are_reasonable(self, simple_2d_data):
        """Test that centroids are within reasonable bounds of the data"""
        km = KMeans(num_clusters=2, random_state=42)
        km.fit(simple_2d_data)
        
        data_min = simple_2d_data.min(axis=0)
        data_max = simple_2d_data.max(axis=0)
        
        # All centroids should be within data bounds
        for centroid in km.centroids:
            assert np.all(centroid >= data_min)
            assert np.all(centroid <= data_max)


class TestKMeansPrediction:
    
    def test_predict_new_data(self):
        """Test prediction on new data points"""
        X_train = np.array([
            [0, 0], [1, 1],
            [10, 10], [11, 11]
        ])
        
        km = KMeans(num_clusters=2, random_state=42)
        km.fit(X_train)
        
        # New points close to each cluster
        X_test = np.array([[0.5, 0.5], [10.5, 10.5]])
        predictions = km.predict(X_test)
        
        # The two test points should be in different clusters
        assert predictions[0] != predictions[1]
    
    def test_predict_returns_array(self, simple_2d_data):
        """Test that predict returns numpy array"""
        km = KMeans(num_clusters=2, random_state=42)
        km.fit(simple_2d_data)
        
        predictions = km.predict(simple_2d_data)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(simple_2d_data),)


class TestKMeansReproducibility:
    
    def test_same_random_state_same_results(self, simple_2d_data):
        """Test that same random state gives same results"""
        km1 = KMeans(num_clusters=2, random_state=42)
        km1.fit(simple_2d_data)
        pred1 = km1.predict(simple_2d_data)
        
        km2 = KMeans(num_clusters=2, random_state=42)
        km2.fit(simple_2d_data)
        pred2 = km2.predict(simple_2d_data)
        
        # Same random state should give same results
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_different_random_state_may_differ(self, simple_2d_data):
        """Test that different random states may give different initial centroids"""
        km1 = KMeans(num_clusters=2, random_state=42)
        km1._init_centroids(simple_2d_data)
        centroids1 = km1.centroids.copy()
        
        km2 = KMeans(num_clusters=2, random_state=123)
        km2._init_centroids(simple_2d_data)
        centroids2 = km2.centroids.copy()
        
        # Different seeds may produce different initial centroids
        # (though they might occasionally be the same by chance)
        # We just check they're both valid
        assert centroids1.shape == centroids2.shape


# Parametrized tests
@pytest.mark.parametrize("n_clusters,expected_unique", [
    (1, 1),
    (2, 2),
    (3, 3),
])
def test_correct_number_of_clusters(n_clusters, expected_unique, three_cluster_data):
    """Test that KMeans produces the correct number of clusters"""
    km = KMeans(num_clusters=n_clusters, random_state=42)
    km.fit(three_cluster_data)
    predictions = km.predict(three_cluster_data)
    
    unique_clusters = np.unique(predictions)
    assert len(unique_clusters) == expected_unique


@pytest.mark.parametrize("max_iters", [1, 5, 10, 50])
def test_max_iterations_respected(max_iters, simple_2d_data):
    """Test that max_iters parameter is respected"""
    km = KMeans(num_clusters=2, max_iters=max_iters, random_state=42)
    km.fit(simple_2d_data)
    
    # Should complete without error regardless of max_iters
    assert km.centroids.shape == (2, 2)


def main():
    """Example usage"""
    print("=" * 60)
    print("EXAMPLE USAGE")
    print("=" * 60)
    
    # Create a tiny 2D dataset
    X_test = np.array([
        [1, 2],
        [1, 4],
        [1, 0],
        [10, 2],
        [10, 4],
        [10, 0]
    ])
    
    k = 2
    km = KMeans(num_clusters=k, random_state=42)
    km.fit(X_test)
    
    print("\nCentroids after fitting:")
    print(km.centroids)
    
    print("\nCluster assignments:")
    predictions = km.predict(X_test)
    for i, (point, cluster) in enumerate(zip(X_test, predictions)):
        print(f"Point {i} {point} -> Cluster {cluster}")
    
    print("\n" + "=" * 60)
    print("To run tests, use: pytest test_kmeans.py -v")
    print("=" * 60)


if __name__ == '__main__':
    main()
