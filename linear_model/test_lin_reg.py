import numpy as np
import pytest
from .linear_regression import LinearRegression


@pytest.fixture
def model():
    return LinearRegression()


def test_fit_and_predict_shapes(model):
    X = np.random.randn(10, 3)
    y = np.random.randn(10)

    model.fit(X, y)
    preds = model.predict(X)

    assert preds.ndim == 1
    assert preds.shape[0] == X.shape[0]


def test_predict_before_fit_raises(model):
    X = np.random.randn(5, 2)

    with pytest.raises(Exception):
        model.predict(X)


def test_exact_linear_fit(model):
    rng = np.random.default_rng(0)

    X = rng.normal(size=(100, 2))
    true_w = np.array([2.0, -3.0])
    y = X @ true_w

    model.fit(X, y)
    preds = model.predict(X)

    np.testing.assert_allclose(preds, y, atol=1e-6)


def test_single_feature(model):
    X = np.arange(10).reshape(-1, 1)
    y = 3 * X.squeeze() + 1

    model.fit(X, y)
    preds = model.predict(X)

    np.testing.assert_allclose(preds, y, atol=1e-6)


def test_predict_is_deterministic(model):
    X = np.random.randn(20, 4)
    y = np.random.randn(20)

    model.fit(X, y)
    preds1 = model.predict(X)
    preds2 = model.predict(X)

    np.testing.assert_array_equal(preds1, preds2)


def test_predict_shape_mismatch(model):
    X_train = np.random.randn(10, 3)
    y = np.random.randn(10)
    X_bad = np.random.randn(5, 2)

    model.fit(X_train, y)

    with pytest.raises(ValueError):
        model.predict(X_bad)


def test_more_samples_than_features(model):
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (100,)
